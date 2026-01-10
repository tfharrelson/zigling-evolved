const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Linear = @import("linear.zig").Linear;
const DSilu = @import("tensor.zig").DSilu;
const Silu = @import("tensor.zig").Silu;
const Index = @import("tensor.zig").Index;
const Allocator = std.mem.Allocator;

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();
        ptr: *anyopaque,
        vtable: *const VTable,

        const VTable = struct {
            forward: *const fn (ptr: *anyopaque, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T),
            backward: *const fn (ptr: *anyopaque, alloc: Allocator, tensor: *Tensor(T), learning_rate: T) TensorError!Tensor(T),
        };

        pub fn init(ptr: anytype) Self {
            const typ = @TypeOf(ptr);
            const ptr_info = @typeInfo(typ);
            std.debug.assert(ptr_info == .pointer); // must be a pointer

            const gen = struct {
                pub fn forward(pointy: *anyopaque, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
                    const self: typ = @ptrCast(@alignCast(pointy));
                    return ptr_info.pointer.child.forward(self, alloc, tensor);
                }
                pub fn backward(pointy: *anyopaque, alloc: Allocator, tensor: *Tensor(T), learning_rate: T) TensorError!Tensor(T) {
                    const self: typ = @ptrCast(@alignCast(pointy));
                    return ptr_info.pointer.child.backward(self, alloc, tensor, learning_rate);
                }
            };

            return .{
                .ptr = ptr,
                .vtable = &.{
                    .forward = gen.forward,
                    .backward = gen.backward,
                },
            };
        }

        pub fn forward(self: Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            return self.vtable.forward(self.ptr, alloc, tensor);
        }

        pub fn backward(self: Self, alloc: Allocator, tensor: *Tensor(T), learning_rate: T) TensorError!Tensor(T) {
            return self.vtable.backward(self.ptr, alloc, tensor, learning_rate);
        }
    };
}

// simple implementation of a model
// TODO: there may be a simpler world where everything is adheres to the model interface
// not seeing it yet, but should look into it. further explanation: there can exist a world
// where a model depends on another model, which means we can have many layers of models
// all within one struct. this probably exists as a sequential model, so should think through
// whether or not this exists as other model architectures - what exists that separates sequential
// models from others? e.g. standard transformers are another variant of sequential models.
pub fn FullyConnectedLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Cache = struct {
            grad: Tensor(T),
            input: Tensor(T),
        };
        linear: Linear(T),
        // TODO: allow for generic activation functions
        store_grad: bool = true,
        cache: ?Cache = null,

        pub fn forward(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            var intermediate = try self.linear.forward(alloc, tensor);
            if (self.store_grad) {
                self.cache = .{
                    .grad = try DSilu(T, alloc, &intermediate),
                    .input = tensor.*,
                };
            }
            return try Silu(T, alloc, &intermediate);
        }

        pub fn backward(self: *Self, alloc: Allocator, tensor: *Tensor(T), learning_rate: T) TensorError!Tensor(T) {
            if (self.cache) |*c| {
                // update the weights automatically? pytorch separates backward from step
                // but i'm having trouble envisioning a situation where those should be decoupled
                // so i'm tentatively going to couple them by default here and regret some mistake later

                // need to check whether or not the caller input a tensor batch
                // if they did, then we have to map over the batch dimension and outer product each
                if (tensor.shape.len == 1) {
                    try tensor.unsqueeze(alloc, 0);
                    try c.input.unsqueeze(alloc, 0);
                }
                if (tensor.shape[0] != c.input.shape[0]) {
                    return TensorError.IncompatibleShapeError;
                }

                // outer product - so unsqueeze edges
                // the first dimension is always the batch index now
                // and we want to keep it that way, so unsqueeze the first non-batch index
                try tensor.unsqueeze(alloc, 1);
                try c.input.unsqueeze(alloc, -1);
                var total_weight_grad: Tensor(T) = try .zeros(alloc, self.linear.shape);
                for (0..tensor.shape[0]) |i| {
                    var indices = [_]Index{ .{ .int = i }, .{ .all = {} }, .{ .all = {} } };
                    var t = try tensor.get(alloc, &indices);
                    var ci = try c.input.get(alloc, &indices);

                    var weight_grad = try ci.matmul(alloc, &t);
                    weight_grad.mul(learning_rate);
                    try total_weight_grad.add(&weight_grad);
                }
                try self.linear.params.add(&total_weight_grad);
                var delta = try tensor.matmul(alloc, &self.linear.params);
                // TODO: make an element-wise multiplication method on tensor
                // TODO: figure out how to manipulate the tensor object directly to save memory
                for (0..c.grad.items.len) |i| {
                    delta.items[i] *= c.grad.items[i];
                }
                return delta;
            } else {
                // TODO: come up with a sensible error handling strategy for this one
                // it's not quite a tensor error so i have to merge tensor errors with layer errors
                return TensorError.UnexpectedError;
            }
        }

        pub fn model(self: *Self) Model(T) {
            return Model(T).init(self);
        }
    };
}

test "fcl model forward happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(f32).init(&elements, &shape);
    const l = try Linear(f32).init(allocator, &shape, null);
    var m = FullyConnectedLayer(f32){ .linear = l };
    const output = try m.model().forward(allocator, &t);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
}

test "fcl model backward happy path" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 1, 2, 3, 4 };
    var shape = [_]usize{ 2, 2 };

    var t = try Tensor(f32).init(&elements, &shape);
    const l = try Linear(f32).init(allocator, &shape, null);
    var m = FullyConnectedLayer(f32){ .linear = l };
    _ = try m.model().forward(allocator, &t);
    const output = try m.model().backward(allocator, &t, 0.01);
    const exp_shape = [_]usize{ 2, 2 };
    for (exp_shape, output.shape) |e, s| {
        try std.testing.expect(e == s);
    }
}
