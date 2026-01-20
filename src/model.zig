const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Linear = @import("linear.zig").Linear;
const DSilu = @import("tensor.zig").DSilu;
const Silu = @import("tensor.zig").Silu;
const Index = @import("tensor.zig").Index;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;

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

                // start by calculating delta from the input tensor, which i call the 'weighted delta tensor'
                // for convenience sake this is equivalent to W_l+1 \dot delta_l+1
                // this way i don't have to pass the weight params backward to this function
                // TODO: make an element-wise multiplication method on tensor
                // TODO: figure out how to manipulate the tensor object directly to save memory
                for (0..c.grad.items.len) |i| {
                    // now the tensor object is delta after this transformation
                    // TODO: do i want to manipulate tensor in place here? will it be needed elsewhere?
                    tensor.items[i] *= c.grad.items[i];
                }
                const weighted_delta = try tensor.matmul(alloc, &self.linear.params);

                // outer product - so unsqueeze edges
                // the first dimension is always the batch index now
                // and we want to keep it that way, so unsqueeze the first non-batch index
                try tensor.unsqueeze(alloc, -1);
                defer tensor.squeeze(alloc, -1) catch @panic("tensor can't be squeezed back to original shape!");
                try c.input.unsqueeze(alloc, 1);
                var total_weight_grad: Tensor(T) = try .zeros(alloc, self.linear.shape);
                for (0..tensor.shape[0]) |i| {
                    var indices = [_]Index{ .{ .int = i }, .{ .all = {} }, .{ .all = {} } };
                    var t = try tensor.get(alloc, &indices);
                    var ci = try c.input.get(alloc, &indices);

                    var weight_grad = try t.matmul(alloc, &ci);
                    weight_grad.mul(learning_rate);
                    try total_weight_grad.add(&weight_grad);
                }
                try self.linear.params.add(&total_weight_grad);

                // now need to multiply the tensor which represents delta right now by the weight matrix of this model
                // to yield the weighted delta tensor that can get passed backwards to other layers
                return weighted_delta;
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
    try expect(exp_shape.len == output.shape.len);
    for (exp_shape, output.shape) |e, s| {
        try expect(e == s);
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
    try expect(exp_shape.len == output.shape.len);
    for (exp_shape, output.shape) |e, s| {
        try expect(e == s);
    }
}

test "fcl model backward correct numerics" {
    const allocator = std.heap.page_allocator;
    var elements = [_]f32{ 10, -10 };
    var shape = [_]usize{2};
    var l_shape = [_]usize{ 2, 2 };

    var t = try Tensor(f32).init(&elements, &shape);
    var l = try Linear(f32).init(allocator, &l_shape, null);
    // use the identity matrix for simplicity
    var test_items = [_]f32{ 1, 0, 0, 1 };
    l.params.items = &test_items;
    var m = FullyConnectedLayer(f32){ .linear = l };
    var res = try m.forward(allocator, &t);
    const backprop = try m.backward(allocator, &res, 0.1);
    try expect(backprop.items[0] < 10.01 and backprop.items[0] > 9.99);
    try expect(backprop.items[1] < 0.01 and backprop.items[1] > -0.01);

    // expected gradient is [[100, -100], [0, 0]] - learning rate is 0.1
    // so we should be adding [10, -10, 0, 0] to the initial params
    const exp_params = [_]f32{ 11, -10, 0, 1 };
    try expect(exp_params.len == m.linear.params.items.len);
    for (exp_params, m.linear.params.items) |e, i| {
        try expect(i - e < 0.01 and i - e > -0.01);
    }
}
