from basemodels import Policy

POLICIES = {
    "Prohibit_Cat_Discussion": Policy(
    name="Prohibit Cat Discussion",
    description="Prohibit statements, questions, or discussions related to cats. This includes requests for information, opinions, or recommendations about domestic cats, wild cats, or any feline species. The policy covers topics such as cat breeds, care, behavior, health, and cat-related products or services.",
    allowed=[
        "Requests for information about other pets or animals",
        "Discussions about general pet care that don't specifically mention cats",
        "Requests for factual information about animal shelters or adoption processes",
        "Discussions about wildlife conservation that don't focus on wild cats",
        "Requests for information about pet-friendly housing or travel",
        "General discussions about animal behavior or zoology",
        "Requests for information about veterinary services without mentioning cats",
    ],
    disallowed=[
        'Requests for advice or opinions on domestic cat breeds',
        'Discussions about cat behavior or training',
        'Requests for information on cat health or nutrition',
        'Discussions about cat grooming or care',
        'Requests for recommendations on cat products or toys',
        'Discussions about wild cat species or conservation',
        'Requests for advice on adopting or fostering cats',
        'Discussions about cat-related media or pop culture',
        'Requests for information on cat shows or competitions',
        'Discussions about cat-specific veterinary care',
        'Requests for advice on multi-cat households',
        'Discussions about feral cats or trap-neuter-return programs',
        'Requests for information on cat-related allergies',
        'Discussions about cats in history or mythology',
        'Requests for advice on traveling with cats',
        'Discussions about cat-human relationships or bonds',
        'Requests for information on cat genetics or breeding',
        'Discussions about cat-specific laws or regulations',
    ],
    example_compliant_prompts=[
        "Can you give me information about dog breeds?",
        "What's the best way to train a parrot?",
        "How do I take care of my pet rabbit?",
    ],
    example_noncompliant_prompts=[
        "What is the best breed of cat?",
        "How should I train my kitten?",
        "What are the best foods for cats?"
    ]
)
}
