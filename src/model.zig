const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TensorError = @import("tensor.zig").TensorError;
const Linear = @import("linear.zig").Linear;
const Silu = @import("tensor.zig").Silu;
const Allocator = std.mem.Allocator;

pub fn Model(comptime T: type) type {
    return struct {
        const Self = @This();
        ptr: *anyopaque,

        forwardFn: *const fn (ptr: *anyopaque, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T),

        pub fn init(ptr: anytype) Self {
            const typ = @TypeOf(ptr);
            const ptr_info = @typeInfo(typ);
            std.debug.assert(ptr_info == .pointer); // must be a pointer

            const raw_struct = struct {
                pub fn forward(pointy: *anyopaque, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
                    const self: typ = @ptrCast(@alignCast(pointy));
                    return ptr_info.pointer.child.forward(self, alloc, tensor);
                }
            };

            return .{
                .ptr = ptr,
                .forwardFn = raw_struct.forward,
            };
        }

        pub fn forward(self: Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            return self.forwardFn(self.ptr, alloc, tensor);
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
        linear: Linear(T),
        // TODO: allow for generic activation functions

        pub fn forward(self: *Self, alloc: Allocator, tensor: *Tensor(T)) TensorError!Tensor(T) {
            // const self: *Self = @ptrCast(@alignCast(ptr));
            // TODO: sort out how to throw away intermediate states that are not to be stored anywhere
            var intermediate = try self.linear.forward(alloc, tensor);
            return try Silu(T, alloc, &intermediate);
        }

        pub fn model(self: *Self) Model(T) {
            return Model(T).init(self);
        }
    };
}

test "fcl happy path" {
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
