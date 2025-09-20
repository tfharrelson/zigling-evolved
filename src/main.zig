const std = @import("std");
const zigling_evolved = @import("zigling_evolved");
const api = @import("proto/SC2APIProtocol.pb.zig");
const protobuf = @import("protobuf");
const ws = @import("websocket");

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
    errdefer _ = c.kill() catch unreachable;

    // start a game using the protobuf api
    var sock = try connect_to_game(allocator, 0);
    defer sock.deinit();

    // TODO: change this to an array list once i do the painstaking chore
    // of moving all managed lists in zig-protobuf to unmanaged array lists
    var players = std.array_list.Managed(api.PlayerSetup).init(allocator);
    defer players.deinit();

    try players.append(api.PlayerSetup{
        .type = api.PlayerType.Participant,
        .race = api.Race.Zerg,
    });
    try players.append(api.PlayerSetup{
        .type = api.PlayerType.Computer,
        .race = api.Race.Zerg,
    });

    const map_dir = try std.process.getEnvVarOwned(allocator, "SC2_MAP_DIR");
    const map_path: []const u8 = try std.fmt.allocPrint(allocator, "{s}/AutomatonLE.SC2Map", .{map_dir});
    const map = api.RequestCreateGame.Map_union{ .local_map = .{ .map_path = protobuf.ManagedString{ .Const = map_path } } };
    const start_game_request: api.Request = .{ .request = .{ .create_game = .{ .player_setup = players, .Map = map } } };
    defer start_game_request.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();
    const req_buf = try start_game_request.encode(arena_alloc);
    try sock.write(req_buf);
    std.debug.print("wrote to server\n", .{});

    // come up with a more explicit way to handle a null message
    const message = try sock.read();
    const resp = try api.Response.decode(message.?.data, allocator);
    if (resp.response) |vresp| {
        switch (vresp) {
            .create_game => std.debug.print("found create game response! error details: {any}", .{vresp.create_game.error_details}),
            else => std.debug.print("found something weird", .{}),
        }
    }

    // TODO: add other processes here like the 2 bots
    _ = try c.wait();
}

fn connect_to_game(allocator: std.mem.Allocator, attempt_num: u8) !ws.Client {
    var arena = std.heap.ArenaAllocator.init(allocator);
    const arena_alloc = arena.allocator();

    var client = ws.Client.init(arena_alloc, .{
        .host = sc2_host,
        .port = sc2_port,
    }) catch |err| {
        if (attempt_num < 5) {
            std.Thread.sleep(2000000000);
            return connect_to_game(allocator, attempt_num + 1);
        } else {
            return err;
        }
    };
    std.debug.print("Connected!\n", .{});

    client.handshake("/sc2api", .{ .timeout_ms = 1000 }) catch |err| {
        if (attempt_num < 5) {
            std.Thread.sleep(2000000000);
            return connect_to_game(allocator, attempt_num + 1);
        } else {
            return err;
        }
    };
    errdefer client.deinit();
    std.debug.print("handshake succeeded.\n", .{});
    return client;
}

test "simple test" {
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(std.testing.allocator); // Try commenting this out and see if zig detects the memory leak!
    try list.append(std.testing.allocator, 42);
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
