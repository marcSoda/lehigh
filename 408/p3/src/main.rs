use std::env;
use std::path::Path;

use phylo;

fn main() -> phylo::Result<()> {
    let args: Vec<String> = env::args().collect(); // Collects command line arguments into a vector of Strings
    if args.len() < 3 {
        println!("Usage: phylo <input> <upgma || neighbor>");
        std::process::exit(1);
    }
    let path = Path::new(&args[1]);
    let method = &args[2];
    let dm = phylo::Matrix::from_file(path)?;
    let newick = match method.as_str() {
        "upgma" => phylo::upgma(&dm),
        "neighbor" => phylo::neighbor(&dm),
        _ => {
            println!("Invalid method. Must be 'upgma' or 'neighbor'");
            std::process::exit(1);
        }
    };
    println!("{}", newick.to_string());
    Ok(())
}
