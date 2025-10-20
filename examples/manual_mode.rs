use wc_fir::{manual_apply, ManualProfile};

fn main() {
    let revenue = vec![100.0, 110.0, 125.0, 140.0, 150.0, 160.0];
    let production = vec![80.0, 82.0, 90.0, 96.0, 100.0, 104.0];

    let profiles = vec![
        ManualProfile {
            percentages: vec![0.6, 0.3, 0.1],
            scale: 0.9,
        },
        ManualProfile {
            percentages: vec![0.7, 0.3],
            scale: 0.4,
        },
    ];

    let synthetic = manual_apply(&[revenue, production], &profiles).unwrap();

    println!("Manual FIR output:");
    for (period, value) in synthetic.iter().enumerate() {
        println!("  t={period}: {value:.3}");
    }
}
