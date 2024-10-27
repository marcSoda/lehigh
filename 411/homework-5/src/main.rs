use std::collections::HashMap;

fn main() {
    let data = vec![
        ("Alice", 50),
        ("Bob", 80),
        ("Charlie", 60),
        ("David", 75),
        ("Eve", 90),
        ("Frank", 55),
        ("Grace", 45),
        ("Hannah", 85),
        ("Ivy", 95),
        ("Jack", 40)
    ];
    let remarks = vec!["Good", "Excellent", "Average", "Good", "Outstanding", "Fair", "Poor", "Very Good", "Exceptional", "Poor"];
    let additional_data = vec![
        ("Kyle", 65),
        ("Liam", 70)
    ];
    let activities = vec![
        vec!["Read", "Write", "Code"],
        vec!["Draw", "Paint"],
        vec!["Code", "Debug"],
        vec!["Paint", "Sculpt"],
        vec!["Read", "Research"],
        vec!["Write", "Blog"],
        vec!["Code", "Review"],
        vec!["Draw", "Design"],
        vec!["Research", "Experiment", "Code"],
        vec!["Sculpt", "Model"],
    ];

    println!("p1:");
    let p1r = p1(data.clone());
    p1r.iter().for_each(|&name| {
        println!("\t{}", name);
    });

    println!("p2:");
    let p2r = p2(data.clone());
    p2r.iter().for_each(|val| {
        println!("\t{}", val);
    });

    println!("p3:");
    let p3r = p3(data.clone(), remarks);
    p3r.iter().for_each(|val| {
        println!("\t{}", val);
    });

    println!("p4:");
    println!("\t{}", p4(data.clone(), additional_data));

    println!("p5:");
    let p5r = p5(data.clone());
    p5r.iter().for_each(|&(name, score)| {
        println!("\t{}: {}", name, score);
    });

    println!("p6:");
    let p6r = p6(data.clone());
    p6r.iter().for_each(|&(name, score)| {
        println!("\t{}: {}", name, score);
    });

    println!("p7:");
    let p7r = p7(data.clone());
    println!("\tAll above 40? {}\n\tAny above 90? {}", p7r.0, p7r.1);

    println!("p8:");
    let p8r = p8(data.clone(), activities.clone());
    p8r.iter().for_each(|(name, acts)| {
        println!("\t{}: {:?}", name, acts);
    });

    println!("p9:");
    let p9r = p9(data.clone(), activities.clone());
    p9r.iter().for_each(|activity| {
        println!("\t{}", activity);
    });

    println!("p10:");
    let p10r = p10(data.clone(), activities.clone());
    p10r.iter().for_each(|(name, acts)| {
        println!("\t{}: {:?}", name, acts);
    });

    println!("p11:");
    let p11r = p11(data.clone());
    p11r.iter().for_each(|&(name, score)| {
        println!("\t{}: {}", name, score);
    });

    println!("p12:");
    let p12_result = p12(data.clone(), activities.clone());
    p12_result.iter().for_each(|(id, (name, score, activities))| {
        println!("\tID: {}:\n\t\tName: {}\n\t\tScore: {}\n\t\tActivities: {:?}", id, name, score, activities);
    });

    println!("p13:");
    let p13r = p13(data.clone(), activities.clone());
    p13r.iter().for_each(|(name, score)| {
        println!("\t{}: {}", name, score);
    });
}

fn p1(data: Vec<(&str, i32)>) -> Vec<&str> {
    data.iter()
        .filter(|&(_, score)| *score > 70)
        .map(|&(name, _)| name)
        .collect::<Vec<_>>()
}

fn p2(data: Vec<(&str, i32)>) -> Vec<String> {
    let mut ret = vec![];
    data.iter().enumerate().for_each(|(index, &(name, score))| {
        ret.push(format!("{}: {} - {}", index, name, score));
    });
    ret
}

fn p3(data: Vec<(&str, i32)>, remarks: Vec<&str>) -> Vec<String> {
    data.iter()
        .zip(remarks.iter())
        .filter_map(|(&(name, score), remark)| {
            if score > 75 { Some(format!("{} - {} ", name, remark)) }
            else { None }
        }).collect::<Vec<_>>()
}

fn p4(data: Vec<(&str, i32)>, additional_data: Vec<(&str, i32)>) -> usize {
    data.iter().chain(additional_data.iter()).count()
}

fn p5(data: Vec<(&str, i32)>) -> Vec<(&str, i32)> {
    let nested_data = vec![vec![data[0], data[1]], vec![data[2], data[3]], vec![data[4], data[5]]];
    nested_data.into_iter()
               .flat_map(|v| v.into_iter())
               .take(5)
               .collect::<Vec<_>>()
}

fn p6(data: Vec<(&str, i32)>) -> Vec<(&str, i32)> {
    data.iter()
        .cloned()
        .cycle()
        .take(15)
        .collect::<Vec<_>>()
}

fn p7(data: Vec<(&str, i32)>) -> (bool, bool) {
    let all_above_40 = data.iter().all(|&(_, score)| score > 40);
    let any_above_90 = data.iter().any(|&(_, score)| score > 90);
    (all_above_40, any_above_90)
}

fn p8(data: Vec<(&str, i32)>, activities: Vec<Vec<&str>>) -> Vec<(String, Vec<String>)> {
    data.iter()
        .zip(activities.iter())
        .filter(|&((_, score), _)| *score > 60)
        .map(|(&(name, _), activities)| {
            (name.to_string(), activities.iter().map(|&activity| activity.to_string()).collect())
        })
        .collect()
}

fn p9(data: Vec<(&str, i32)>, activities: Vec<Vec<&str>>) -> Vec<String> {
    data.iter()
        .zip(activities.iter())
        .filter(|&((_, score), _)| *score > 70)
        .flat_map(|(_, activities)| activities.iter())
        .map(|&activity| activity.to_string())
        .fold(Vec::new(), |mut acc, activity| {
            if !acc.contains(&activity) { acc.push(activity); }
            acc
        })
}

fn p10(data: Vec<(&str, i32)>, activities: Vec<Vec<&str>>) -> Vec<(String, Vec<String>)> {
    let mut found = false;
    data.iter()
        .zip(activities.iter())
        .scan((), |_, ((name, score), activities)| {
            if found { None }
            else {
                if *score < 50 { found = true; }
                Some((name.to_string(), activities.iter().map(|&activity| activity.to_string()).collect()))
            }
        }).collect()
}

fn p11(data: Vec<(&str, i32)>) -> Vec<(&str, i32)> {
    data.into_iter()
        .skip_while(|&(_, score)| score >= 60)
        .take_while(|&(_, score)| score <= 90)
        .fuse()
        .collect()
}

// I'm not using chain because that makes no sense
fn p12(data: Vec<(&str, i32)>, activities: Vec<Vec<&str>>) -> HashMap<usize, (String, i32, Vec<String>)> {
    data.into_iter()
        .zip(activities.into_iter())
        .enumerate()
        .map(|(id, ((name, score), activities))| {
            (id, (name.to_string(), score, activities.into_iter().map(String::from).collect()))
        })
        .collect()
}

fn p13(data: Vec<(&str, i32)>, activities: Vec<Vec<&str>>) -> Vec<(String, i32)> {
    data.into_iter()
        .zip(activities.into_iter())
        .filter(|&((_, score), ref activities)| score > 65 && activities.contains(&"Code"))
        .map(|((name, score), _)| (name.to_string(), score))
        .collect()
}
