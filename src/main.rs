use std::{collections::HashMap, error::Error, fmt};
use statrs::distribution::{ContinuousCDF, Normal};

#[derive(Debug, serde::Deserialize)]
struct Record {
    #[serde(rename = "Date")]
    date: String,
    #[serde(rename = "Close")]
    close: f64,
}

struct Interval {
    start: f64,
    end: f64,
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let r_i_values = csv::Reader::from_path("MSFT.csv")?
        .deserialize()
        .collect::<Result<Vec<Record>, _>>()?
        .windows(2)
        .map(|window| (window[1].close / window[0].close).ln())
        .collect::<Vec<f64>>();

    let categories = vec![
        Interval { start: f64::NEG_INFINITY, end: -0.001        },
        Interval { start: -0.001           , end: -0.0004       },
        Interval { start: -0.0004          , end: 0.0           },
        Interval { start: 0.0              , end: 0.0004        },
        Interval { start: 0.0004           , end: 0.001         },
        Interval { start: 0.001            , end: f64::INFINITY },
    ];

    let distribution = Normal::new(0.0, 0.02).unwrap();
    println!("Question 3(a):");
    solve(&r_i_values, &categories, &distribution);
    println!("");

    let n = r_i_values.len();
    let sample_mean = r_i_values.iter().sum::<f64>() / n as f64;
    let sample_variance = r_i_values.iter().map(|r_i| (r_i - sample_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let distribution = Normal::new(sample_mean, sample_variance.sqrt()).unwrap();
    println!("Question 3(b):");
    solve(&r_i_values, &categories, &distribution);

    Ok(())
}

fn solve(r_i_values: &Vec<f64>, categories: &Vec<Interval>, distribution: &Normal) {
    let n = r_i_values.len();
    let mut observed = HashMap::new();

    for r_i in r_i_values.iter() {
        for category in categories.iter() {
            if *r_i >= category.start && *r_i < category.end {
                *observed.entry(category.to_string()).or_insert(0) += 1;
                break;
            }
        }
    }

    let mut expected = HashMap::new();
    let mut test_statistic = HashMap::new();

    for category in categories.iter() {
        let p = distribution.cdf(category.end) - distribution.cdf(category.start);
        let expected_value = p * n as f64;
        expected.insert(category.to_string(), expected_value);
        let observed_value = f64::from(*(observed.get(&category.to_string()).unwrap()));
        test_statistic.insert(category.to_string(), (observed_value - expected_value).powi(2) / expected_value);
    }

    for category in categories.iter() {
        let key = category.to_string();
        let observed_value = observed.get(&key).unwrap();
        let expected_value = expected.get(&key).unwrap();
        let test_statistic_value = test_statistic.get(&key).unwrap();
        println!("{}:\n {}, {:.3}, {:.3}\n", key, observed_value, expected_value, test_statistic_value);
    }

    println!("Test statistic: {:.3}", test_statistic.values().sum::<f64>());
}