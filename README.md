# Machine Learning Language Models for Chemical Generation

### Note: project in early stages and under active development. This README is out of date almost immediately upon writing. 

This repository contains code and notebooks for training and using machine learning language models to generate chemical structures, specifically SMILES strings. The models are designed to learn the syntax and semantics of chemical representations, enabling the generation of novel molecules.

Most language modelling tools are aimed at natural language processing, but the chemical language of SMILES strings is a rich and complex language in its own right, with syntax, grammar and vocabulary all of its own. This repository provides tools for training and using models to generate SMILES strings, which can be used to explore the chemical space and generate novel molecules.

## Application Example: Generating Caffeine-Like molecules with an LSTM.

**Note, out of date. Correct way to do this now is to condition initial hidden state of model to produce molecules with desired properties.**

We could say that we hope to generate _variants_ of an existing molecule, but with random, physically plausible changes to the structure.

Lets take caffiene, every scientists favourite, without it no work would get done. It is described by the following SMILES string `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`.

We can generate a series of caffiene-like molecules by providing the model with a portion of the caffiene SMILES and asking it to predict the next character in the SMILES string.


```python
n_chars = 100

n_valid = 0
n_invalid = 0
images = []
for i in range(20):
    output = simple_rnn.simple_generate("CN1C=NC2=C1C(=O)N", n_chars, model, chembl.char_to_idx, chembl.idx_to_char, temperature=0.8, device='cuda')
    print(f"Requested up to {n_chars} characters, got: {len(output)}")
    print(f"Molecule Canonical SMILES: {output}")

    valid = utilities.validate_smiles_string(output)

    if not valid:
        n_invalid += 1
        print("Generated SMILES is not valid.")
    else:
        n_valid += 1
        print("Generated SMILES is valid.")
        images.append(utilities.draw_molecule(output))
```

    Requested up to 100 characters, got: 30
    Molecule Canonical SMILES: CN1C=NC2=C1C(=O)N(CCN)C2=CC2=O
    Generated SMILES is not valid.


    [09:35:59] SMILES Parse Error: unclosed ring for input: 'CN1C=NC2=C1C(=O)N(CCN)C2=CC2=O'


    Requested up to 100 characters, got: 96
    Molecule Canonical SMILES: CN1C=NC2=C1C(=O)N(CC(=O)N[C@H](CC(=O)Nc3ccc(OCc4ccccc4)cc3)c3ccccc3)C(=O)N1C=C2C(=O)c2ccccc2C1=O
    Generated SMILES is valid.
    Requested up to 100 characters, got: 19
    Molecule Canonical SMILES: CN1C=NC2=C1C(=O)N13
    Generated SMILES is not valid.

    [OUTPUT CURTAILED]


```python
display(Markdown(f"# Generated {n_valid} valid caffiene-like molecules and {n_invalid} invalid SMILES strings out of 20 attempts."))
display(Markdown("## Generated Molecules"))
for img in images:
    display(img)
```


## Generated 10 valid caffeine-like molecules and 10 invalid SMILES strings out of 20 attempts.

Here is the structure of caffeiene for reference:

![caffeine](readme_img/caffeine_reference.png)

Below, we can see examples of some of the caffeine-like molecules generated with our model. The model has learned to generate molecules which are structurally similar to caffeine, but with variations in the side chains and functional groups. Some of these molecules are quite complex, with multiple rings and functional groups, while others are simpler. 

## Sample of Generated Molecules
   
![png](readme_img/character_level_rnn_generator_16_2.png)
![png](readme_img/character_level_rnn_generator_16_3.png)
![png](readme_img/character_level_rnn_generator_16_4.png)
![png](readme_img/character_level_rnn_generator_16_5.png)
![png](readme_img/character_level_rnn_generator_16_6.png)
![png](readme_img/character_level_rnn_generator_16_7.png)   
![png](readme_img/character_level_rnn_generator_16_8.png)
    
For such a simple model, the results of this test are rather impressive. The model has learned to generate valid SMILES strings, which correspond to real molecules. In order to do this, the model will have implicitly "learned" the correct valences for atoms, common functional groups, and the rules of SMILES syntax, which is not a trivial task.

We can see that the model has learned to generate a variety of different molecules, some of which are quite complex. The model is able to generate molecules with rings, branches, and various functional groups, all while adhering to the rules of SMILES syntax. Occasionally included are rarer functional groups which are common in some pharmaceuticals, such as trifluoromethyl groups (CF3).

Of course, because SMILES strings are syntactially rigid, often with long-range dependencies, a simple model like this smaller LSTM will sometimes generate molecules which are _almost_ but not quite valid. Models become better at avoiding these syntactic errors with longer training and larger models. 

Above, we can see that the model has learned to generate valid SMILES strings, but we can also see information on the types of errors the model makes. These fall into two categories, syntactic errors, where the model generates a string which is not valid SMILES, and semantic errors, where the model generates a valid SMILES string but one which does not correspond to a real molecule. 

Syntactic: e.g. The model has a tendancy to open parentheses but not close them, or opening rings but not specifying where they close:

`SMILES Parse Error: extra open parentheses while parsing: CC(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCC
SMILES Parse Error: check for mistakes around position 96:
(CC(C)C)C(=O)N[C@@H](CCCC
`

Semantic: E.g. Occasionally generating molecules where atoms have incorrect valences (e.g. F with two bonds):

`SMILES Parse Error: Failed parsing SMILES ' HC(=O)N[C@H](C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O''
Explicit valence for atom # 0 F, 2, is greater than permitted` 

The model also appears to have a real desire to generate long, saturated hydrocarbon chains. The training data does include some of these, so it's not surprising that the model has learned to generate them. If in doubt, the next character is probably just a saturated carbon atom.
