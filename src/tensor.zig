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
    int: u32,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []T = &[_]T{},
        shape: []usize = &[_]usize{},

        pub const empty: Self = .{ .shape = &[_]usize{}, .items = &[_]T{} };

        pub fn blah(_: usize) u32 {
            return 42;
        }

        pub fn init(self: *Self, items: []T, shape: []usize) !void {
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
            self.shape = shape;
            self.items = items;
            return;
        }

        pub fn get(self: *Self, allocator: std.mem.Allocator, indices: []Index) !Tensor(T) {
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
            var new_tensor: Tensor(T) = .empty;
            new_tensor.init(elements.items, new_shape.items) catch unreachable;
            return new_tensor;
        }

        pub fn matmul(self: *Self, other: Tensor(T)) !Tensor(T) {
            // standard matrix multiplication that takes the last dim
            // of self and multiplies with the first dim of other.
            if (self.shape[self.shape.len - 1] != other.shape[0]) {
                return TensorError.IncompatibleShapeError;
            }
        }
    };
}

test "tensor init" {
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    var t: Tensor(u32) = .empty;
    try t.init(&elements, &shape);
    try expect(@TypeOf(t) == Tensor(u32));
    for (t.items, elements) |item, elem| {
        try expect(item == elem);
    }
    for (t.shape, shape) |s1, s2| {
        try expect(s1 == s2);
    }
    try expect(Tensor(u32).blah(11) == 42);
}

test "tensor init errors" {
    var too_many_elements = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    var shape = [_]usize{ 2, 3 };

    var t: Tensor(u32) = .empty;
    const shape_error = t.init(&too_many_elements, &shape);
    try std.testing.expectError(TensorError.IncompatibleShapeError, shape_error);

    var elements = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var too_many_dimensions: [10]usize = @splat(1);
    const high_dim_error = t.init(&elements, &too_many_dimensions);
    try std.testing.expectError(TensorError.IncompatibleShapeError, high_dim_error);
}

test "tensor get" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    var t: Tensor(u32) = .empty;
    try t.init(&elements, &shape);

    var indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 } };
    const t2 = try t.get(allocator, &indices);
    try expect(t2.items.len == 1);
    try expect(t2.items[0] == 5);
}

test "tensor get all" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    var t: Tensor(u32) = .empty;
    try t.init(&elements, &shape);

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

    var t: Tensor(u32) = .empty;
    try t.init(&elements, &shape);

    var oob_indices = [_]Index{ .{ .int = 4 }, .{ .int = 1 } };
    const oob_error = t.get(allocator, &oob_indices);
    try std.testing.expectError(TensorError.OutOfBounds, oob_error);

    var too_many_indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 }, .{ .int = 0 } };
    const shape_error = t.get(allocator, &too_many_indices);
    try std.testing.expectError(TensorError.IncompatibleShapeError, shape_error);
}
