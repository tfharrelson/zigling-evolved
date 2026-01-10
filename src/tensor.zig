const std = @import("std");
const expect = std.testing.expect;
const math = std.math;
const Allocator = std.mem.Allocator;

pub const TensorError = error{
    IncompatibleShapeError,
    HighDimensionality,
    OutOfBounds,
    // TODO: come up with more rigorous error handling
    // Need this to make a proper interface
    OutOfMemory,
    UnexpectedError,
};

const IndexTag = enum { all, int };

pub const Index = union(IndexTag) {
    all: void,
    int: usize,
};

pub fn Tensor(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []T = undefined,
        shape: []usize = undefined,

        pub const empty: Self = .{ .shape = &[_]usize{}, .items = &[_]T{} };

        pub fn init(items: []T, shape: []usize) TensorError!Self {
            var expected_num_items: usize = 1;
            for (shape) |i| {
                expected_num_items *= i;
            }
            if (shape.len > 10) {
                return TensorError.HighDimensionality;
            }
            if (items.len != expected_num_items) {
                return TensorError.IncompatibleShapeError;
            }
            return .{ .shape = shape, .items = items };
        }

        pub fn zeros(allocator: Allocator, shape: []usize) TensorError!Self {
            if (shape.len > 10) {
                return TensorError.HighDimensionality;
            }
            var size: usize = 1;
            for (shape) |s| {
                size *= s;
            }
            var zeros_list = std.ArrayList(T).initCapacity(allocator, size) catch return TensorError.OutOfMemory;
            for (0..size) |_| {
                zeros_list.append(allocator, 0) catch return TensorError.OutOfMemory;
            }
            return .{ .shape = shape, .items = zeros_list.items };
        }

        pub fn get(self: *const Self, allocator: Allocator, indices: []Index) TensorError!Tensor(T) {
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
                    },
                }
            }

            var inner_idx: u32 = 0;
            var new_shape = std.ArrayList(usize).initCapacity(allocator, dim) catch return TensorError.OutOfMemory;
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
                        new_shape.insertAssumeCapacity(0, self.shape[rev_idx]);
                        // new_shape.append(allocator, self.shape[rev_idx]) catch return TensorError.OutOfMemory;
                        weights.append(allocator, curr_weight) catch return TensorError.OutOfMemory;
                        inner_idx += 1;
                    },
                    IndexTag.int => {
                        offset += curr_weight * indices[rev_idx].int;
                    },
                }
                curr_weight *= self.shape[rev_idx];
            }

            // wait wait wait
            // don't need any weights during the loop, just need to get the indices of the contracted matrix, e.g.
            // if i have a 6x5x4x3x2 tensor and i want a slice over [2,All,0,All,1], then I just need the indices
            // within the 5x3 inner matrix. The algorithm for that is much much simpler. It's a two step using
            // modulo and floordivide. i % N_1 = index_1, floorDivide(i, N_1) => i, loop over the dimensionality
            // of the inner matrix. Then once we get all indices, apply weights to each index value, sum them up
            // add the offset and you have an index in the high dimensional `elements` array on self.
            // NOTE: the weights list is already in reverse, the new_shape list is in the correct orientation
            // TODO: should try to figure out how to clean up the reverse vs non-reverse logic in this part.
            // good news though, we have decent tests for making this a bit safer
            var elements: std.ArrayList(T) = .empty;
            for (0..num_elems) |i| {
                var elem_idx: usize = 0;
                var floor_idx: usize = i;
                for (0..new_shape.items.len) |j| {
                    const rev_idx = new_shape.items.len - 1 - j;
                    // elem_idx += @mod(i, new_shape.items[rev_idx]) * weights.items[rev_idx];
                    // floor_idx = @divFloor(floor_idx, new_shape.items[rev_idx]);
                    elem_idx += @mod(floor_idx, new_shape.items[rev_idx]) * weights.items[j];
                    floor_idx = @divFloor(floor_idx, new_shape.items[rev_idx]);
                }
                elements.append(allocator, self.items[elem_idx + offset]) catch return TensorError.OutOfMemory;
            }
            return Tensor(T).init(elements.items, new_shape.items) catch unreachable;
        }

        pub fn matmul(self: *Self, allocator: Allocator, other: *Tensor(T)) TensorError!Tensor(T) {
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
            for (other.shape[1..other.shape.len]) |s| {
                other_num_items *= s;
            }

            // got to loop through possible index combinations and calculate dot product for each
            var output_elements = std.ArrayList(T).initCapacity(allocator, self_num_items * other_num_items) catch return TensorError.OutOfMemory;
            for (0..self_num_items) |i| {
                for (0..other_num_items) |j| {
                    var self_index_list = try Tensor(T).getIndexList(allocator, i, self.shape[0 .. self.shape.len - 1]);
                    // TODO: handle the single dimension case more gracefully
                    var rest_other_index_list: std.ArrayList(Index) = .empty;
                    if (other.shape.len != 1) {
                        rest_other_index_list = try Tensor(T).getIndexList(allocator, j, other.shape[1..other.shape.len]);
                    }
                    self_index_list.append(allocator, .{ .all = {} }) catch return TensorError.OutOfMemory;
                    var other_index_list: std.ArrayList(Index) = .empty;
                    other_index_list.append(allocator, .{ .all = {} }) catch return TensorError.OutOfMemory;
                    other_index_list.appendSlice(allocator, rest_other_index_list.items) catch return TensorError.OutOfMemory;

                    // meat of calculation - convert to simd and calculate dot product
                    var value: T = 0;
                    const row = try self.get(allocator, self_index_list.items);
                    const column = try other.get(allocator, other_index_list.items);
                    for (row.items, column.items) |self_item, other_item| {
                        value += self_item * other_item;
                    }
                    output_elements.append(allocator, value) catch return TensorError.OutOfMemory;
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
            new_shape_list.appendSlice(allocator, self.shape[0 .. self.shape.len - 1]) catch return TensorError.OutOfMemory;
            new_shape_list.appendSlice(allocator, other.shape[1..]) catch return TensorError.OutOfMemory;
            return Tensor(T).init(output_elements.items, new_shape_list.items) catch unreachable;
        }

        fn getIndexList(allocator: Allocator, element_idx: usize, shape: []usize) TensorError!std.ArrayList(Index) {
            // get the array indices of a specific element index in the 'items' list
            var output_list: std.ArrayList(Index) = .empty;
            // can't i just: for elem 11 in a 4x5 matrix
            // 11 floor 5 = 2 check
            // (11 - 10) floor 1 = 1 check
            // for elem 3 in a 2x2 matrix
            // 3 floor 2 = 1 check
            // 3 - 2*1 floor 1 = 1 check
            var size: usize = 1;
            for (shape) |s| {
                size *= s;
            }
            var remaining_elems = element_idx;
            for (shape) |s| {
                size /= s;
                const index = @divFloor(remaining_elems, size);
                // const index = @mod(remaining_elems, s);
                output_list.append(allocator, .{ .int = index }) catch return TensorError.OutOfMemory;
                remaining_elems -= index * size;
            }
            return output_list;
        }

        pub fn argmax(self: *Self) TensorError!usize {
            // TODO: generalize this to use any shape - i have a feeling this may be necessary at some point
            if (self.shape.len != 1) {
                return TensorError.IncompatibleShapeError;
            }
            var max_value: T = self.items[0];
            var max_index: usize = 0;
            for (1..self.items.len) |idx| {
                if (self.items[idx] > max_value) {
                    max_value = self.items[idx];
                    max_index = idx;
                }
            }
            return max_index;
        }

        pub fn unsqueeze(self: *Self, alloc: Allocator, dim: isize) TensorError!void {
            const udim = @abs(dim);
            const slice_dim = if (dim < 0) self.shape.len - udim + 1 else udim;
            const left: []usize = self.shape[0..slice_dim];
            const right: []usize = self.shape[slice_dim..];
            var new_shape = std.ArrayList(usize).initCapacity(alloc, self.shape.len + 1) catch return TensorError.OutOfMemory;
            new_shape.appendSlice(alloc, left) catch return TensorError.OutOfMemory;
            new_shape.append(alloc, 1) catch return TensorError.OutOfMemory;
            new_shape.appendSlice(alloc, right) catch return TensorError.OutOfMemory;
            self.shape = new_shape.items;
        }

        pub fn mul(self: *Self, scalar: T) void {
            // scalar multiplication that mutates self
            for (0..self.items.len) |i| {
                self.items[i] = self.items[i] * scalar;
            }
        }

        pub fn add(self: *Self, other: *Tensor(T)) TensorError!void {
            // tensor addition between compatible shapes that mutates self
            for (self.shape, other.shape) |ss, os| {
                if (ss != os) {
                    return TensorError.IncompatibleShapeError;
                }
            }
            for (0..self.items.len) |i| {
                self.items[i] += other.items[i];
            }
        }
    };
}

pub fn Silu(comptime T: type, allocator: Allocator, t: *Tensor(T)) TensorError!Tensor(T) {
    // TODO: revisit the allocation semantics - it would be way more efficient to activate
    // tensor values in place. but right now it's just easier to allocate more memory for every op
    // just to get things working.
    var activated_items: std.ArrayList(T) = .empty;
    for (t.items) |i| {
        activated_items.append(allocator, i * sigmoid(T, i)) catch return TensorError.OutOfMemory;
    }
    return Tensor(T).init(activated_items.items, t.shape) catch unreachable;
}

inline fn sigmoid(comptime T: type, val: T) T {
    return 1 / (1 + math.exp(-val));
}

pub fn DSilu(comptime T: type, allocator: Allocator, t: *Tensor(T)) TensorError!Tensor(T) {
    // derivative of the silu function
    var derivative_items: std.ArrayList(T) = .empty;
    var s: T = undefined;
    for (t.items) |i| {
        s = sigmoid(T, i);
        derivative_items.append(allocator, s + i * (s * (1 - s))) catch return TensorError.OutOfMemory;
    }
    return Tensor(T).init(derivative_items.items, t.shape) catch unreachable;
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

    var t = try Tensor(u32).init(&elements, &shape);

    var indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 } };
    const t2 = try t.get(allocator, &indices);
    try expect(t2.items.len == 1);
    try expect(t2.items[0] == 5);
}

test "tensor get all" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    var t = try Tensor(u32).init(&elements, &shape);

    var indices = [_]Index{ .{ .all = {} }, .{ .int = 1 } };
    const t2 = try t.get(allocator, &indices);
    try expect(t2.items.len == 2);
    try expect(t2.items[0] == 2);
    try expect(t2.items[1] == 5);
}

test "tensor get all 3d" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 1, 2, 3 };

    var t = try Tensor(u32).init(&elements, &shape);
    var indices = [_]Index{ .{ .int = 0 }, .{ .all = {} }, .{ .all = {} } };
    const t2 = try t.get(allocator, &indices);
    for (0..t2.shape.len) |i| {
        try expect(t2.shape[i] == t.shape[i + 1]);
    }
    for (0..elements.len) |i| {
        try expect(t2.items[i] == elements[i]);
    }
}

test "tensor get errors" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4, 5, 6 };
    var shape = [_]usize{ 2, 3 };

    var t = try Tensor(u32).init(&elements, &shape);

    var oob_indices = [_]Index{ .{ .int = 4 }, .{ .int = 1 } };
    const oob_error = t.get(allocator, &oob_indices);
    try std.testing.expectError(TensorError.OutOfBounds, oob_error);

    var too_many_indices = [_]Index{ .{ .int = 1 }, .{ .int = 1 }, .{ .int = 0 } };
    const shape_error = t.get(allocator, &too_many_indices);
    try std.testing.expectError(TensorError.IncompatibleShapeError, shape_error);
}

test "tensor get index list 3d" {
    const allocator = std.heap.page_allocator;

    var shape = [_]usize{ 1, 2, 2 };
    const index_list = try Tensor(u32).getIndexList(allocator, 3, &shape);
    const exp_items = [_]Index{ .{ .int = 0 }, .{ .int = 1 }, .{ .int = 1 } };
    try expect(index_list.items.len == exp_items.len);
    for (index_list.items, exp_items) |i, e| {
        expect(i.int == e.int) catch {
            std.debug.print("failed, index_list = {any}, expected = {any}\n", .{ index_list, exp_items });
            return error.TestUnexpectedError;
        };
    }
}

test "tensor matmul happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(u32).init(&elements, &shape);

    const output = try t.matmul(allocator, &t);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 7, 10, 15, 22 };
    for (exp_items, output.items) |e, i| {
        try std.testing.expect(e == i);
    }
}

test "tensor matmul vectors" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2 };
    var shape1 = [_]usize{ 1, 2 };
    var shape2 = [_]usize{ 2, 1 };

    var t1 = try Tensor(u32).init(&elements, &shape1);
    var t2 = try Tensor(u32).init(&elements, &shape2);

    const output = try t2.matmul(allocator, &t1);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 1, 2, 2, 4 };
    for (exp_items, output.items) |e, i| {
        try std.testing.expect(e == i);
    }
}

test "tensor silu" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ -10, 10 };
    var shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &shape);

    const activated_t = try Silu(f32, allocator, &t);

    try expect(activated_t.shape.len == 1);
    try expect(activated_t.items.len == 2);

    try expect(activated_t.items[0] < 0 and activated_t.items[0] > -0.01);
    try expect(activated_t.items[1] < 10 and activated_t.items[1] > 9.99);
}

test "tensor dsilu" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ -10, 10 };
    var shape = [_]usize{2};

    var t = try Tensor(f32).init(&elements, &shape);

    const derivative_t = try DSilu(f32, allocator, &t);

    try expect(derivative_t.shape.len == 1);
    try expect(derivative_t.items.len == 2);

    try expect(derivative_t.items[0] < 0 and derivative_t.items[0] > -0.01);
    try expect(derivative_t.items[1] < 1.01 and derivative_t.items[1] > 0.99);
}

test "tensor argmax" {
    var elements = [_]f32{ -10, 10, 0.4 };
    var shape = [_]usize{3};

    var t = try Tensor(f32).init(&elements, &shape);

    const res = try t.argmax();
    try expect(res == 1);
}

test "tensor unsqueeze left" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(u32).init(&elements, &shape);

    try t.unsqueeze(allocator, 0);
    const exp_shape = [_]usize{ 1, 2, 2 };
    for (exp_shape, t.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 1, 2, 3, 4 };
    for (exp_items, t.items) |e, i| {
        try std.testing.expect(e == i);
    }
}

test "tensor unsqueeze right" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(u32).init(&elements, &shape);

    try t.unsqueeze(allocator, -1);
    const exp_shape = [_]usize{ 2, 2, 1 };
    for (exp_shape, t.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 1, 2, 3, 4 };
    for (exp_items, t.items) |e, i| {
        try std.testing.expect(e == i);
    }
}

test "tensor unsqueeze middle" {
    const allocator = std.heap.page_allocator;
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(u32).init(&elements, &shape);

    try t.unsqueeze(allocator, 1);
    const exp_shape = [_]usize{ 2, 1, 2 };
    for (exp_shape, t.shape) |e, s| {
        try std.testing.expect(e == s);
    }
    const exp_items = [_]u32{ 1, 2, 3, 4 };
    for (exp_items, t.items) |e, i| {
        try std.testing.expect(e == i);
    }
}

test "tensor scalar mul" {
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(u32).init(&elements, &shape);
    t.mul(2);
    const exp_elems = [_]u32{ 2, 4, 6, 8 };
    for (exp_elems, t.items) |e, i| {
        try expect(e == i);
    }
}

test "tensor add happy path" {
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t1 = try Tensor(u32).init(&elements, &shape);

    var elements2 = [_]u32{ 2, 3, 4, 5 };
    var t2 = try Tensor(u32).init(&elements2, &shape);
    try t1.add(&t2);
    const exp_elems = [_]u32{ 3, 5, 7, 9 };
    for (exp_elems, t1.items) |e, i| {
        try expect(e == i);
    }
}

test "tensor add shape error" {
    var elements = [_]u32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t1 = try Tensor(u32).init(&elements, &shape);

    var elements2 = [_]u32{ 2, 3, 4, 5, 6, 7 };
    var shape2 = [_]usize{ 2, 3 };
    var t2 = try Tensor(u32).init(&elements2, &shape2);
    const res = t1.add(&t2);

    try std.testing.expectError(TensorError.IncompatibleShapeError, res);
}
