enum Route {
    Local(u16),
    Express,
}

fn fare(route: Route) -> u16 {
    // Express rides use a flat fare.
    match route {
        Route::Local(stops) => 2 + stops.min(5),
        Route::Express => 9,
    }
}

fn main() {
    let rides = [Route::Local(3), Route::Express];
    for ride in rides {
        println!("fare {}", fare(ride));
    }
}
