# Changelog

<!--next-version-placeholder-->

## v0.5.5 (2024-01-26)

### Fix

* **deps:** Update dependency lightgbm to v4.3.0 ([#136](https://github.com/34j/boost-loss/issues/136)) ([`7e3f49a`](https://github.com/34j/boost-loss/commit/7e3f49a1d970f2efac53cef73ff771b99d093583))

## v0.5.4 (2023-12-31)

### Fix

* **deps:** Update dependency attrs to v23.2.0 ([#130](https://github.com/34j/boost-loss/issues/130)) ([`7620a33`](https://github.com/34j/boost-loss/commit/7620a335fb761bd22c983cddefa1e4266d126980))

## v0.5.3 (2023-12-22)

### Fix

* **deps:** Update dependency lightgbm to v4.2.0 ([#127](https://github.com/34j/boost-loss/issues/127)) ([`f090ad3`](https://github.com/34j/boost-loss/commit/f090ad3437c52e1a1227bad045580220f75b6e82))

## v0.5.2 (2023-12-20)

### Fix

* **deps:** Update dependency xgboost to v2.0.3 ([#126](https://github.com/34j/boost-loss/issues/126)) ([`b4ddf21`](https://github.com/34j/boost-loss/commit/b4ddf2157670a82bbcb79ee5e58188de777d5bd5))

## v0.5.1 (2023-11-13)

### Fix

* **deps:** Update dependency xgboost to v2.0.2 ([#111](https://github.com/34j/boost-loss/issues/111)) ([`9f52c01`](https://github.com/34j/boost-loss/commit/9f52c01fb93bb38b507539ebb3160ea0a75ada99))

## v0.5.0 (2023-11-07)

### Feature

* **sklearn:** Add `copy_loss`, `apply_objective` and `apply_eval_metric` parameter to `apply_custom_loss()` ([#108](https://github.com/34j/boost-loss/issues/108)) ([`e877604`](https://github.com/34j/boost-loss/commit/e8776041eb08082c99d00500987edcb75184cbeb))

## v0.4.2 (2023-11-05)

### Fix

* **sklearn:** Fix `_recursively_set_random_state()` imcompatible with `EstimatorWrapperBase` ([#107](https://github.com/34j/boost-loss/issues/107)) ([`ed13289`](https://github.com/34j/boost-loss/commit/ed13289e71dbc74ce5cc814eddfafd397db0e271))

## v0.4.1 (2023-11-05)

### Fix

* **sklearn:** Check if estimator has `get_params()` and `set_params()` in `apply_custom_loss()` ([#106](https://github.com/34j/boost-loss/issues/106)) ([`3b77017`](https://github.com/34j/boost-loss/commit/3b77017c00a421a58fee5f725eae36975c422a92))

## v0.4.0 (2023-11-04)

### Feature

* Fix `recursive_strict` behavior for `apply_custom_loss` and `patch`, set default `var_type` to `"std"`, add more parameters to `VarianceEstimator` ([#105](https://github.com/34j/boost-loss/issues/105)) ([`24e90ac`](https://github.com/34j/boost-loss/commit/24e90ac96758ceebf665f2c74fd1e1d839d42534))

## v0.3.6 (2023-10-24)

### Fix

* **deps:** Update dependency xgboost to v2.0.1 ([#102](https://github.com/34j/boost-loss/issues/102)) ([`3c7040d`](https://github.com/34j/boost-loss/commit/3c7040d0e73838c282ddf7fbeac728e6f4534066))

## v0.3.5 (2023-10-24)

### Fix

* **sklearn:** Add `return_std` parameter in `VarianceEstimator` ([#101](https://github.com/34j/boost-loss/issues/101)) ([`d7c1625`](https://github.com/34j/boost-loss/commit/d7c16255483f87fe0f81b84bb393a0aefa4774d8))

## v0.3.4 (2023-10-23)

### Fix

* **deps:** Update dependency lightgbm-callbacks to v0.1.5 ([#99](https://github.com/34j/boost-loss/issues/99)) ([`b655a82`](https://github.com/34j/boost-loss/commit/b655a8259e3988e8ca2633c0feec1cc128a485d4))

## v0.3.3 (2023-10-15)

### Fix

* **deps:** Update dependency lightgbm to v4 ([#40](https://github.com/34j/boost-loss/issues/40)) ([`adb35ed`](https://github.com/34j/boost-loss/commit/adb35ed53f2eea7de53996f07e6d6ac043727f41))

## v0.3.2 (2023-10-11)

### Fix

* **deps:** Update dependency lightgbm-callbacks to v0.1.4 ([#88](https://github.com/34j/boost-loss/issues/88)) ([`d96cf5e`](https://github.com/34j/boost-loss/commit/d96cf5e54fa7297e5eac6e7622a398098aa8891f))

## v0.3.1 (2023-10-11)

### Fix

* **deps:** Update dependency xgboost to v2 ([#70](https://github.com/34j/boost-loss/issues/70)) ([`856bdbb`](https://github.com/34j/boost-loss/commit/856bdbb5233bc0cdb0d5a9e0e0d07110bbd664d1))

## v0.3.0 (2023-10-11)

### Feature

* **sklearn:** Add `recursive_strict` parameter to `patch()` ([#86](https://github.com/34j/boost-loss/issues/86)) ([`677e40f`](https://github.com/34j/boost-loss/commit/677e40f29136365e96c8aa22f50e53d64562d92a))

## v0.2.2 (2023-09-18)

### Fix

* **deps:** Update dependency catboost to v1.2.1 ([#58](https://github.com/34j/boost-loss/issues/58)) ([`61159e2`](https://github.com/34j/boost-loss/commit/61159e2b4ad444c8f045c2f5c83ac9eb3531f178))

## v0.2.1 (2023-09-18)

### Fix

* **deps:** Update dependency lightgbm-callbacks to v0.1.2 ([#74](https://github.com/34j/boost-loss/issues/74)) ([`7871101`](https://github.com/34j/boost-loss/commit/7871101b6552e4a0395883197e3d9207951ef8e2))

## v0.2.0 (2023-06-17)

### Feature

* **sklearn:** Add utility function `patch()` ([#24](https://github.com/34j/boost-loss/issues/24)) ([`4436256`](https://github.com/34j/boost-loss/commit/4436256cbc1be66daa2e4dfadccc4325c7b5fc7c))

## v0.1.1 (2023-06-16)

### Fix

* Fix `patch_catboost()` and `patch_ngboost()` ([#23](https://github.com/34j/boost-loss/issues/23)) ([`9a5c061`](https://github.com/34j/boost-loss/commit/9a5c0618453b82d5cef8d1adb9df66e3568084d8))

## v0.1.0 (2023-05-28)
### Feature
* Add main feat ([#1](https://github.com/34j/boost-loss/issues/1)) ([`a70d977`](https://github.com/34j/boost-loss/commit/a70d97710524ec6b33773474e6cccdb8dfa55909))

### Documentation
* Add @34j as a contributor ([`5974bf4`](https://github.com/34j/boost-loss/commit/5974bf4d243c577f44839cff17fa2732c54c0dba))
