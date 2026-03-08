const std = @import("std");
const expect = std.testing.expect;
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Allocator = std.mem.Allocator;

pub fn Linear(comptime T: type) type {
    return struct {
        const Self = @This();

        shape: *[2]usize,
        params: Tensor(T),

        pub fn init(alloc: Allocator, shape: *[2]usize, seed: ?usize) TensorError!Self {
            var random_seed: usize = undefined;
            if (seed) |s| {
                random_seed = s;
            } else {
                std.posix.getrandom(std.mem.asBytes(&random_seed)) catch return TensorError.UnexpectedError;
            }
            var pnrg = std.Random.DefaultPrng.init(random_seed);
            const rand = pnrg.random();
            var num_elems: usize = 1;
            for (shape) |s| {
                num_elems *= s;
            }
            var arr = std.ArrayList(T).initCapacity(alloc, num_elems) catch return TensorError.OutOfMemory;
            for (0..num_elems) |_| {
                arr.append(alloc, rand.float(T)) catch return TensorError.OutOfMemory;
            }
            const params = try Tensor(T).init(arr.items, shape);
            return Self{ .shape = shape, .params = params };
        }

        pub fn forward(self: *Self, alloc: Allocator, state: *Tensor(T)) TensorError!Tensor(T) {
            // NOTE: apply the transpose of the operation to work seamlessly with batch dimension
            // This will likely get confusing as I tend to think of the input tensor being to the
            // right of the the parameter tensors when writing this out on paper. May have to introduce
            // some way to view the model parameters in the more standard state. Calculations should
            // remain in this direction as it makes way too much sense. It's either that or, the
            // batch dimension of the input tensor has to be the last index of the shape, which is
            // also weird.
            return try state.matmul(alloc, &self.params);
        }
    };
}

test "linear matmul happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(f32).init(&elements, &shape);
    var l = try Linear(f32).init(allocator, &shape, null);

    const output = try l.forward(allocator, &t);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
}
