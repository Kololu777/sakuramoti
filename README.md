# Sakuramoti is optical flow and point tracking library for pytorch framework

Sakuramoti(sakuramoti) will support various recent optical-flow and point-tracking model modules. In addition, relevant utilities will be implemented to facilitate the development and study of optical-flow and point-tracking models.

---

# TODO List
##  Code
### Model
- [ ] WIP: RAFT
  - [x] Done core model code
  - [x] WIP: Demo code
  - [x] WIP: Tool code
  - [x] WIP: Test code
  - [ ] Alternate Corr Block
- [ ] Pips
  - [x] Done Refactor
  - [ ] WIP: Test code
    - [x] Smoke
    - [x] Pretrained integrate
    - [ ] Evaluate Check
      - I Confirmed slight differences between the original implementation and the visualization results. so, currently recnfirm cause.
  - [x] WIP: Pretrain integrate
- [ ] TAPIR

### Loss & Metrics
- [x] Sequence Loss
  - [x] Loss function & class code
  - [x] Test code
- [x] BCE Loss
-
- [x] epe
  - [x] Metrics code
  - [x] Test code
- [ ] WIP: TAP-Net Metric
  - [ ] Convert numpy to pytorch.
  - [ ] Test Code
### Type Guard
- [ ] Tensor

## etc
- [ ] WIP: Ruff+mypy CI
- [ ] document
- [ ] enviroment tool
