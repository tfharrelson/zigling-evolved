const std = @import("std");
const zigling_evolved = @import("zigling_evolved");
const api = @import("proto/SC2APIProtocol.pb.zig");

const sc2_host = "0.0.0.0";
const sc2_port = 5000;
pub fn main() !void {
    // Prints to stderr, ignoring potential errors.
    std.debug.print("All your {s} are belong to us.\n", .{"codebase"});
    try zigling_evolved.advancedPrint();

    const allocator = std.heap.page_allocator;

    const cmd = try std.process.getEnvVarOwned(allocator, "SC2_EXE");
    // TODO: parametrize port and url
    var buf: [4]u8 = undefined;
    const port_string = try std.fmt.bufPrint(&buf, "{}", .{sc2_port});
    const argv = [_][]const u8{ cmd, "-listen", sc2_host, "-port", port_string };
    var c = std.process.Child.init(&argv, allocator);
    try c.spawn();

    // start a game using the protobuf api
    _ = try connect_to_game(allocator);

    // TODO: add other processes here like the 2 bots
    _ = try c.wait();
}

fn connect_to_game(allocator: std.mem.Allocator) !*std.http.Client.Connection {
    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();
    var client = std.http.Client{ .allocator = arena_alloc };
    // const headers = &[_]std.http.Header{
    //     .{.name = "Content-Type", .value = "application/octet-stream"}
    // };
    var attempt_num: u8 = 0;
    std.debug.print("Connecting to game\n", .{});
    while (attempt_num < 5) {
        if (client.connect(sc2_host, sc2_port, std.http.Client.Connection.Protocol.plain)) |conn| {
            std.debug.print("Connected!\n", .{});
            return conn;
        } else |err| switch (err) {
            std.http.Client.ConnectError.ConnectionRefused => {
                attempt_num += 1;
                std.time.sleep(2000000000);
            },
            else => return err,
        }
    }
    return std.http.Client.ConnectError.ConnectionRefused;
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
