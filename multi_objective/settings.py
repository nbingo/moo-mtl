
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[1, 1],
    epochs=50,
    use_scheduler=False,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[1, 1],
    epochs=50,
    use_scheduler=False,
)

compas = dict(
    dataset='compas',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[1, 1],
    epochs=50,
    use_scheduler=False,
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    scheduler_milestones=[15,30,45,60,75,90],
)


multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    scheduler_milestones=[15,30,45,60,75,90],
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=[22, 39],   # list(range(40)) for all tasks
    objectives=['BinaryCrossEntropyLoss', 'BinaryCrossEntropyLoss'],
    # task_ids=list(range(40)),
    # objectives=['BinaryCrossEntropyLoss' for _ in range(40)],
    reference_point=[2, 2],
)

paretoMTL = dict(
    method='ParetoMTL',
    lr=1e-3,
    batch_size=256,
    epochs=5,
    num_starts=5,
)

afeature = dict(
    method='afeature',
    lr=1e-3,
    batch_size=256,
    epochs=150,
    num_starts=1,
    early_fusion=True,
    late_fusion=False,
    alpha_generator_dim=2,
    alpha_dir=None,   # dirichlet sampling, None=Uniform sampling
)

SingleTaskSolver = dict(
    method='SingleTask',
    lr=1e-4,
    batch_size=256,
    epochs=100,
    num_starts=1,
    task=0,
)

hyperSolver = dict(
    method='hyper',
    lr=1e-4,
    batch_size=256,
    epochs=100,     # 100 for multi_mnist
    num_starts=1,
    alpha_dir=.2,   # dirichlet sampling
)

generic = dict(
    logdir='results',
    num_workers=4,  # dataloader worker threads
    n_test_rays=25,
    eval_every=1,
    train_eval_every=1, # 0 for not evaluating on the train set
    use_scheduler=True,
)
