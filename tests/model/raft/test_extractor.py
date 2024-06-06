from sakuramoti.flow_model.raft import ResidualBlock

model = ResidualBlock(64, 128, 'group', 1)

print(model)