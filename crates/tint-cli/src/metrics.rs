use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tint_core::SyntaxClass;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub confusion_matrix: [[u64; 9]; 9],
    pub non_whitespace_total: u64,
    pub whitespace_excluded: u64,
    pub ambiguous_excluded: u64,
}

impl Counts {
    pub fn observe(
        &mut self,
        whitespace: bool,
        truth: Option<SyntaxClass>,
        prediction: SyntaxClass,
    ) {
        if whitespace {
            self.whitespace_excluded += 1;
            return;
        }
        self.non_whitespace_total += 1;
        if let Some(truth) = truth {
            self.confusion_matrix[truth.id()][prediction.id()] += 1;
        } else {
            self.ambiguous_excluded += 1;
        }
    }

    pub fn finish(self) -> Metrics {
        let mut classes = Vec::new();
        let mut correct = 0;
        let mut total = 0;
        for class in SyntaxClass::ALL {
            let id = class.id();
            let support = self.confusion_matrix[id].iter().sum::<u64>();
            let predicted = self.confusion_matrix.iter().map(|row| row[id]).sum::<u64>();
            let tp = self.confusion_matrix[id][id];
            correct += tp;
            total += support;
            classes.push(ClassMetrics {
                class,
                support,
                precision: ratio(tp, predicted),
                recall: ratio(tp, support),
                f1: ratio(2 * tp, support + predicted),
            });
        }
        let supported = classes.iter().filter(|class| class.support > 0).count();
        let macro_f1 = if supported == 0 {
            0.0
        } else {
            classes
                .iter()
                .filter(|class| class.support > 0)
                .map(|class| class.f1)
                .sum::<f64>()
                / supported as f64
        };
        Metrics {
            counts: self,
            classes,
            evaluated_tokens: total,
            agreement: ratio(correct, total),
            macro_f1,
        }
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassMetrics {
    pub class: SyntaxClass,
    pub support: u64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub counts: Counts,
    pub classes: Vec<ClassMetrics>,
    pub evaluated_tokens: u64,
    pub agreement: f64,
    pub macro_f1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub overall: Metrics,
    pub per_language: BTreeMap<String, Metrics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metrics_use_supported_truth_classes_and_exclude_unscored_tokens() {
        let mut counts = Counts::default();
        counts.observe(false, Some(SyntaxClass::Keyword), SyntaxClass::Keyword);
        counts.observe(false, Some(SyntaxClass::Keyword), SyntaxClass::Plain);
        counts.observe(false, None, SyntaxClass::Plain);
        counts.observe(true, None, SyntaxClass::Plain);
        let metrics = counts.finish();
        assert_eq!(metrics.agreement, 0.5);
        assert!((metrics.macro_f1 - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(metrics.classes[4].precision, 1.0);
        assert_eq!(metrics.classes[4].recall, 0.5);
        assert_eq!(metrics.classes[4].support, 2);
        assert_eq!(metrics.counts.non_whitespace_total, 3);
        assert_eq!(metrics.counts.whitespace_excluded, 1);
        assert_eq!(metrics.counts.ambiguous_excluded, 1);
        assert_eq!(Counts::default().finish().macro_f1, 0.0);
    }
}
