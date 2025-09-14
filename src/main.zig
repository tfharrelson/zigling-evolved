const std = @import("std");
const zigling_evolved = @import("zigling_evolved");

pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try zigling_evolved.advancedPrint();

    const allocator = std.heap.page_allocator;

    const cmd = try std.process.getEnvVarOwned(allocator, "SC2_EXE");
    // TODO: parametrize port and url
    const argv = [_][]const u8{ cmd, "-listen", "0.0.0.0", "-port", "5000" };
    var c = std.process.Child.init(&argv, allocator);
    try c.spawn();
    // start a game using the protobuf api

    // TODO: add other processes here like the 2 bots
    _ = try c.wait();
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // Try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
