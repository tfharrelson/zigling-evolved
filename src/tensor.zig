const std = @import("std");
const expect = std.testing.expect;

const TensorError = error{
    IncompatibleShapeError,
    HighDimensionality,
    OutOfBounds,
};

const IndexTag = enum { all, int };

const Index = union(IndexTag) {
    all: void,
    int: usize,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []T = &[_]T{},
        shape: []usize = &[_]usize{},

        pub const empty: Self = .{ .shape = &[_]usize{}, .items = &[_]T{} };

        pub fn init(items: []T, shape: []usize) !Self {
            var expected_num_items: usize = 1;
            for (shape) |i| {
                expected_num_items *= i;
            }
            if (shape.len > 10) {
                return TensorError.HighDimensionality;
            }
            if (items.len != expected_num_items) {
                std.debug.print("Unexpected number of items {d} found given shape {d}.\n", .{ items.len, expected_num_items });
                return TensorError.IncompatibleShapeError;
            }
            return .{ .shape = shape, .items = items };
        }

        pub fn get(self: *const Self, allocator: std.mem.Allocator, indices: []Index) !Tensor(T) {
            if (indices.len != self.shape.len) {
                return TensorError.IncompatibleShapeError;
            }

            // get the number of elements of the output tensor
            var num_elems: usize = 1;
            var dim: usize = 0;
            for (self.shape, indices) |s, i| {
                switch (i) {
                    IndexTag.all => {
                        num_elems *= s;
                        dim += 1;
                    },
                    IndexTag.int => |idx| {
                        if (idx >= s) {
                            return TensorError.OutOfBounds;
                        }
                        num_elems *= 1;
                    },
                }
            }

            var inner_idx: u32 = 0;
            var new_shape: std.ArrayList(usize) = .empty;
            var offset: usize = 0;
            var curr_weight: usize = 1;
            var weights: std.ArrayList(usize) = .empty;
            // TODO: swap this for column major pattern, this reversing indexes happens
            // all the time already for basic workflows and it's annoying plus prob has
            // performance issues.
            for (0..indices.len) |i| {
                const rev_idx = indices.len - 1 - i;
                switch (indices[rev_idx]) {
                    IndexTag.all => {
                        try new_shape.append(allocator, self.shape[rev_idx]);
                        try weights.append(allocator, curr_weight);
                        inner_idx += 1;
                    },
                    IndexTag.int => {
                        offset += curr_weight * indices[rev_idx].int;
                    },
                }
                curr_weight *= self.shape[rev_idx];
            }

            var elements: std.ArrayList(T) = .empty;
            for (0..num_elems) |i| {
                var elem_idx: usize = 0;
                var floor_idx: usize = i;
                for (0..new_shape.items.len) |j| {
                    const rev_idx = new_shape.items.len - 1 - j;
                    elem_idx += @mod(i, new_shape.items[rev_idx]) * weights.items[rev_idx];
                    floor_idx = @divFloor(floor_idx, new_shape.items[rev_idx]);
                }
                try elements.append(allocator, self.items[elem_idx + offset]);
            }
            return Tensor(T).init(elements.items, new_shape.items) catch unreachable;
        }

        pub fn matmul(self: *const Self, allocator: std.mem.Allocator, other: Tensor(T)) !Tensor(T) {
            // standard matrix multiplication that takes the last dim
            // of self and multiplies with the first dim of other.
            if (self.shape[self.shape.len - 1] != other.shape[0]) {
                return TensorError.IncompatibleShapeError;
            }
            // find the number of output elements
            var self_num_items: usize = 1;
            for (self.shape[0 .. self.shape.len - 1]) |s| {
                self_num_items *= s;
            }
            var other_num_items: usize = 1;
            for (other.shape[0 .. other.shape.len - 1]) |s| {
                other_num_items *= s;
            }

            // got to loop through possible index combinations and calculate dot product for each
            var output_elements = try std.ArrayList(T).initCapacity(allocator, self_num_items * other_num_items);
            for (0..self_num_items) |i| {
                for (0..other_num_items) |j| {
                    var self_index_list = try Tensor(T).getIndexList(allocator, i, self.shape[0 .. self.shape.len - 1]);
                    const rest_other_index_list = try Tensor(T).getIndexList(allocator, j, other.shape[0 .. other.shape.len - 1]);
                    try self_index_list.append(allocator, .{ .all = {} });
                    var other_index_list: std.ArrayList(Index) = .empty;
                    try other_index_list.append(allocator, .{ .all = {} });
                    try other_index_list.appendSlice(allocator, rest_other_index_list.items);

                    // meat of calculation - convert to simd and calculate dot product
                    var value: T = 0;
                    const row = try self.get(allocator, self_index_list.items);
                    const column = try other.get(allocator, other_index_list.items);
                    for (row.items, column.items) |self_item, other_item| {
                        value += self_item * other_item;
                    }
                    try output_elements.append(allocator, value);
                    // TODO: refactor this to loop over comptime known vector lengths
                    // currently unclear how to handle different element sizes for a fixed simd size
                    // try output_elements.append(allocator, @reduce(.Add, // simd add op
                    //     @as(@Vector(self.shape[self.shape.len - 1], T), self.get(allocator, self_index_list.items).items) //
                    //         * @as(@Vector(other.shape[0], T), other.get(allocator, other_index_list.items).items) //
                    //     ));
                }
            }
            // get new shape
            var new_shape_list: std.ArrayList(usize) = .empty;
            try new_shape_list.appendSlice(allocator, self.shape[0 .. self.shape.len - 1]);
            try new_shape_list.appendSlice(allocator, other.shape[1..]);
            return try Tensor(T).init(output_elements.items, new_shape_list.items); // catch unreachable;
        }

        fn getIndexList(allocator: std.mem.Allocator, element_idx: usize, shape: []usize) !std.ArrayList(Index) {
            // get the array indices of a specific element index in the 'items' list
            var output_list: std.ArrayList(Index) = .empty;
            var size: usize = 1;
            for (shape) |s| {
                size *= s;
            }
            // element 11 in a 4x5 matrix is elem (2, 1)
            // can i do whatever mod math in any order to figure out the index?
            // 11 % 4 = 3 -> x
            // 11 % 5 = 1 -> check
            // 11 - 1 % 4 = 2 -> check
            // so no, the order does matter and has to be done in reverse
            for (0..shape.len) |i| {
                const s = shape[shape.len - 1 - i];
                try output_list.append(allocator, .{ .int = @mod(element_idx, size) });
                size /= s;
            }
            // inplace reverse the list
            // TODO: remove this nonsense once i sort out switching to column major ordering
            // i'm just doing too many reverses for this to make sense
            var left: usize = 0;
            var right: usize = output_list.items.len - 1;
            while (left < right) {
                const left_cache = output_list.items[left];
                output_list.items[left] = output_list.items[right];
                output_list.items[right] = left_cache;
                left += 1;
                right -= 1;
            }
            return output_list;
        }
    };
}

test "tensor init" {
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    const t = try Tensor(u32).init(&elements, &shape);
    try expect(@TypeOf(t) == Tensor(u32));
    for (t.items, elements) |item, elem| {
        try expect(item == elem);
    }
    for (t.shape, shape) |s1, s2| {
        try expect(s1 == s2);
    }
}

test "tensor init errors" {
    var too_many_elements = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    var shape = [_]usize{ 2, 3 };

    const shape_error = Tensor(u32).init(&too_many_elements, &shape);
    try std.testing.expectError(TensorError.IncompatibleShapeError, shape_error);

    var elements = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var too_many_dimensions: [10]usize = @splat(1);
    const high_dim_error = Tensor(u32).init(&elements, &too_many_dimensions);
    try std.testing.expectError(TensorError.IncompatibleShapeError, high_dim_error);
}

test "tensor get" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    const t = try Tensor(u32).init(&elements, &shape);

    var indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 } };
    const t2 = try t.get(allocator, &indices);
    try expect(t2.items.len == 1);
    try expect(t2.items[0] == 5);
}

test "tensor get all" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    const t = try Tensor(u32).init(&elements, &shape);

    var indices = [_]Index{ .{ .all = {} }, .{ .int = 1 } };
    const t2 = try t.get(allocator, &indices);
    try expect(t2.items.len == 2);
    try expect(t2.items[0] == 2);
    try expect(t2.items[1] == 5);
}

test "tensor get errors" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    const t = try Tensor(u32).init(&elements, &shape);

    var oob_indices = [_]Index{ .{ .int = 4 }, .{ .int = 1 } };
    const oob_error = t.get(allocator, &oob_indices);
    try std.testing.expectError(TensorError.OutOfBounds, oob_error);

    var too_many_indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 }, .{ .int = 0 } };
    const shape_error = t.get(allocator, &too_many_indices);
    try std.testing.expectError(TensorError.IncompatibleShapeError, shape_error);
}

test "tensor matmul happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    const t = try Tensor(u32).init(&elements, &shape);

    const output = try t.matmul(allocator, t);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 7, 10, 15, 22 };
    for (exp_items, output.items) |e, i| {
        try std.testing.expect(e == i);
    }
}
