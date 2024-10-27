use h1::{
    array_set::ArraySet,
    hash_set::HashSet,
    list_set::ListSet,
    tree_set::TreeSet,
    Set,
    SetType,
};
use std::time::{Duration, Instant};
use rand::Rng;
use clap::{ Arg, Command };

// Holds command line args
struct Args {
    i: usize,
    k: usize,
    r: usize,
    d: String,
}

fn main() {
    println!("Parsing Args");
    let args = get_args();
    // Get data type from args
    let d: SetType = match args.d.as_str() {
        "array" => SetType::ArraySet,
        "hashset" => SetType::HashSet,
        "list" => SetType::ListSet,
        "tree" => SetType::TreeSet,
        _ => panic!("Invalid Data Structure Requested"),
    };
    if args.r > 100 { panic!("Invalid Ratio Specified") }
    println!("Running");
    // Run test and get time taken
    let elapsed = run(args.i, args.k, args.r, d);
    println!("Operations concluded in {:.4?}", elapsed);
}

// Run test and return time taken
pub fn run(i: usize, k: usize, r: usize, d: SetType) -> Duration {
    // Get prepopulated set
    let mut set = create_populate_set(k, d);
    let mut rng = rand::thread_rng();
    let start = Instant::now();
    // For every operation
    for _ in 0..i {
        // get random number
        let val: usize = rng.gen_range(0..k);
        // random number to determine operation type
        let op_type: usize = rng.gen_range(0..100);

        // r% reads, the rest is split between insert and remove
        if op_type < r {
            set.find(val);
        } else if op_type < (100 + r) / 2 {
            set.insert(val);
        } else {
            set.remove(val);
        }
    }
    start.elapsed()
}

// create and prepopulate the set based on SetType
fn create_populate_set(k: usize, d: SetType) -> Box<dyn Set<usize>> {
    // create set
    let mut set: Box<dyn Set<usize>> = match d {
        SetType::ArraySet => Box::new(ArraySet::new()),
        SetType::HashSet => Box::new(HashSet::new()),
        SetType::ListSet => Box::new(ListSet::new()),
        SetType::TreeSet => Box::new(TreeSet::new()),
    };

    // populate set
    let mut rng = rand::thread_rng();
    for _ in 0..(k / 2) {
        let val: usize = rng.gen_range(0..k);
        set.insert(val);
    }
    set
}

// use clap to get command line args
fn get_args() -> Args {
    let matches = Command::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .subcommand_required(false)
        .arg_required_else_help(true)
        .arg(
            Arg::new("operations")
                .short('i')
                .long("operations")
                .help("number of operations")
                .required(true)
                .value_parser(clap::value_parser!(usize)))
        .arg(
            Arg::new("max key value")
                .short('k')
                .long("max_key")
                .help("max key value")
                .required(true)
                .value_parser(clap::value_parser!(usize)))
        .arg(
            Arg::new("read only ratio")
                .short('r')
                .long("ratio")
                .help("read only ratio")
                .required(true)
                .value_parser(clap::value_parser!(usize)))
        .arg(Arg::new("data structure")
            .short('d')
            .long("structure")
            .help("data structure (array, list, tree, or hashtable)")
            .required(true))
        .get_matches();

    // collect args into Args struct
    Args {
        i: matches.get_one::<usize>("operations").copied().unwrap_or_default(),
        k: matches.get_one::<usize>("max key value").copied().unwrap_or_default(),
        r: matches.get_one::<usize>("read only ratio").copied().unwrap_or_default(),
        d: matches.get_one::<String>("data structure").cloned().unwrap_or_default(),
    }
}
