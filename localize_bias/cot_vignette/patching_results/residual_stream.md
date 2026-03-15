Activation Patching on Residual Stream w/ CoT prompts
Condition Name: Depression
BHC = De-gendered medical notes of a patient w/ depression symptoms 
3 Prompt Types ( A,B,C) x 5 prompt variations each x 1 BHC cases each = 15 runs
5 prompt variations vary only by phrasing and word choice. 
All prompts finish with : “You must start with the following: “Gender: “ “
 15 runs took 10+ hours. 
The patching is applied at every layer, at every token. From here, we get rewrite scores of every token at every layer.Aggregation of Rewrite Scores: 
 I aggregated the rewrite scores across all tokens at the same layer -> Every layer of every prompt type x variation has a single average score  ( In total, 28 layers * 3 prompt types * 5 variations = 420 scores)
Aggregated the rewrite scores across all prompt types and variations at the same layer -> Every layer has a single average score across all prompt types x variations (28 layers = 28 scores)

Identify Layers with Highest Rewrite Scores
Plot 1: Average rewrite score across layers. 





Layers 13, 15 - 18 present  high rewrite scores ( highest is 18) , near 1.0 , for Prompt A and Prompt C. 
Layer 13 presents the highest rewrite score for Prompt B. 

Plot 2: Top 20 Tokens by Rewrite Score ( average across all layers and prompt variations) 


Prompt A and C present more reasonable tokens : depression, patient, hospital, diagnosed. 
Prompt B present some unreasonable tokens: 4, an, the


2. Decode the Tokens of Identified Layers w/ Logit Lens

Layers that present high rewrite scores for prompt A and C ( layer 15 - 18) do not present gender-related or reasonable tokens as observed below. 
Gender-related tokens started to appear only from layer 22nd. See output

I extracted the hidden states at third token of the llm output. LLM output starts with : 
                        Gender: Female 
token pos:         1st    2nd  3rd



W_U shape: torch.Size([152064, 3584])
norm_weight shape: torch.Size([3584])
norm_eps: 1e-06


  PROMPT_C — Logit Lens at Layer 15
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' strugg'  0.1503
       2.               'prech'  0.1007
       3.           'privation'  0.0974
       4.        'WithDuration'  0.0514
       5.           'améliorer'  0.0481
       6.                ' wur'  0.0338
       7.               'isure'  0.0267
       8.                '一个职业'  0.0202
       9.               ' arbe'  0.0108
      10.               'intér'  0.0098
      11.               'traîn'  0.0079
      12.          '.ElementAt'  0.0079
      13.              'enance'  0.0072
      14.                'Grün'  0.0065
      15.                 '的带领'  0.0060
      16.                '>tag'  0.0060
      17.              ' treff'  0.0052
      18.                '(GUI'  0.0035
      19.     '.openConnection'  0.0033
      20.         'precedented'  0.0033


======================================================================
  PROMPT_C — Logit Lens at Layer 16
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' strugg'  0.1302
       2.               'rvine'  0.0837
       3.        'WithDuration'  0.0771
       4.           'améliorer'  0.0576
       5.               'intér'  0.0223
       6.               'isure'  0.0219
       7.           'privation'  0.0158
       8.               'prech'  0.0144
       9.               'traîn'  0.0142
      10.         'precedented'  0.0134
      11.        '.RadioButton'  0.0121
      12.                ' öde'  0.0118
      13.              'achsen'  0.0097
      14.                 '的带领'  0.0093
      15.          '.ElementAt'  0.0082
      16.                'bove'  0.0079
      17.                  '美媒'  0.0075
      18.                ' wur'  0.0074
      19.               ' arbe'  0.0067
      20.               "'icon"  0.0061


======================================================================
  PROMPT_C — Logit Lens at Layer 17
======================================================================

  --- var1 ---
  Position   77token='p'
       1.           'améliorer'  0.1453
       2.             ' strugg'  0.1190
       3.         'precedented'  0.0651
       4.        'WithDuration'  0.0497
       5.               'rvine'  0.0396
       6.           'privation'  0.0324
       7.                   '轫'  0.0302
       8.               'traîn'  0.0289
       9.               'isure'  0.0145
      10.                '.uml'  0.0116
      11.          'paralleled'  0.0096
      12.                  '美媒'  0.0095
      13.         '.TabControl'  0.0070
      14.                'Grün'  0.0069
      15.                   '⇐'  0.0064
      16.              'enance'  0.0058
      17.                '一个职业'  0.0057
      18.                '家都知道'  0.0056
      19.          "';\n\n\n\n"  0.0045
      20.                ' wur'  0.0044


======================================================================
  PROMPT_C — Logit Lens at Layer 18
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' strugg'  0.1463
       2.        'WithDuration'  0.1139
       3.                '一个职业'  0.0535
       4.              'enance'  0.0392
       5.                  '美媒'  0.0268
       6.         '.TabControl'  0.0245
       7.           'améliorer'  0.0227
       8.                   '轫'  0.0189
       9.                   'Ū'  0.0149
      10.                'Grün'  0.0144
      11.         'precedented'  0.0116
      12.                '.uml'  0.0096
      13.                '家都知道'  0.0080
      14.   ' FindObjectOfType'  0.0078
      15.                '特别声明'  0.0063
      16.    'PointerException'  0.0059
      17.                 '在玩家'  0.0058
      18.           'privation'  0.0055
      19.                 '从根本'  0.0050
      20.                 '版权归'  0.0042


======================================================================
  PROMPT_C — Logit Lens at Layer 19
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' strugg'  0.1085
       2.                '特别声明'  0.0645
       3.        'WithDuration'  0.0621
       4.               'Hôtel'  0.0403
       5.           'améliorer'  0.0385
       6.                '一个职业'  0.0344
       7.                  '部副'  0.0287
       8.   ' FindObjectOfType'  0.0197
       9.    'PointerException'  0.0193
      10.                  '性价'  0.0121
      11.                '(GUI'  0.0119
      12.                 '从根本'  0.0108
      13.                   '轫'  0.0091
      14.                'ceği'  0.0080
      15.               "'=>['"  0.0074
      16.                '家都知道'  0.0072
      17.               'traîn'  0.0070
      18.                  '所所'  0.0062
      19.         '.TabControl'  0.0062
      20.         '.isLoggedIn'  0.0062


======================================================================
  PROMPT_C — Logit Lens at Layer 20
======================================================================

  --- var1 ---
  Position   77token='p'
       1.                  '部副'  0.1655
       2.    'PointerException'  0.1145
       3.                  '性价'  0.1018
       4.           'améliorer'  0.0463
       5.                 '的带领'  0.0365
       6.                  '所所'  0.0333
       7.                '家都知道'  0.0328
       8.                '特别声明'  0.0138
       9.                '他是一个'  0.0105
      10.                   '哐'  0.0094
      11.                  '打拼'  0.0085
      12.                 '网首页'  0.0067
      13.                 '毫无疑'  0.0065
      14.                '.uml'  0.0062
      15.              ' intox'  0.0061
      16.               'Hôtel'  0.0056
      17.               "'=>['"  0.0054
      18.                 '力还是'  0.0052
      19.                   '蚱'  0.0051
      20.          '?"\n\n\n\n'  0.0049


======================================================================
  PROMPT_C — Logit Lens at Layer 21
======================================================================

  --- var1 ---
  Position   77token='p'
       1.                '家都知道'  0.2334
       2.                '相关负责'  0.1083
       3.                  '部副'  0.0758
       4.                '他是一个'  0.0220
       5.                  '性价'  0.0213
       6.                '.omg'  0.0180
       7.                'isex'  0.0152
       8.                 '在玩家'  0.0096
       9.        'WithDuration'  0.0085
      10.                  '打拼'  0.0071
      11.                 '力还是'  0.0070
      12.                 '网首页'  0.0068
      13.               'Hôtel'  0.0065
      14.    'PointerException'  0.0064
      15.                 '的带领'  0.0060
      16.                '(GUI'  0.0060
      17.             '-scenes'  0.0058
      18.               "'=>['"  0.0056
      19.                '为您提供'  0.0055
      20.                 '说实话'  0.0048


======================================================================
  PROMPT_C — Logit Lens at Layer 22
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' female'  0.5651
       2.               ' male'  0.2685
       3.                  '女性'  0.0654
       4.             ' gender'  0.0499
       5.                  '男性'  0.0181
       6.                  '性别'  0.0082
       7.            ' females'  0.0068
       8.             ' Female'  0.0050
       9.              ' males'  0.0035
      10.              'Female'  0.0015
      11.              'female'  0.0011
      12.             ' Gender'  0.0011
      13.               ' Male'  0.0011
      14.              'gender'  0.0010
      15.                   '女'  0.0005
      16.                   '男'  0.0005
      17.           ' feminine'  0.0005
      18.            ' genders'  0.0003
      19.                'male'  0.0003
      20.                   '⼥'  0.0003


======================================================================
  PROMPT_C — Logit Lens at Layer 23
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' female'  0.1689
       2.               ' male'  0.0747
       3.             '-scenes'  0.0220
       4.             'steller'  0.0186
       5.                '他是一个'  0.0164
       6.                  '女性'  0.0164
       7.                   '籼'  0.0146
       8.                   '女'  0.0121
       9.                   '蚂'  0.0120
      10.               'Hôtel'  0.0112
      11.                '为您提供'  0.0098
      12.             '.mapbox'  0.0092
      13.                   '贸'  0.0075
      14.          '".\n\n\n\n'  0.0075
      15.                   '⼥'  0.0071
      16.              'Female'  0.0069
      17.                  '不限'  0.0065
      18.            ' patient'  0.0064
      19.             ' Female'  0.0062
      20.                 '最好是'  0.0055


======================================================================
  PROMPT_C — Logit Lens at Layer 24
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' female'  0.2831
       2.                  '不限'  0.0370
       3.               ' male'  0.0240
       4.                 'cis'  0.0219
       5.        ' undisclosed'  0.0168
       6.                   '⼥'  0.0142
       7.          '".\n\n\n\n'  0.0118
       8.                  '女性'  0.0091
       9.                   '女'  0.0087
      10.             '-scenes'  0.0083
      11.             'steller'  0.0082
      12.             ' Female'  0.0068
      13.             '.mapbox'  0.0068
      14.                '他是一个'  0.0058
      15.              'Female'  0.0054
      16.          '?"\n\n\n\n'  0.0045
      17.                   '贸'  0.0044
      18.                 '力还是'  0.0038
      19.        ' unspecified'  0.0037
      20.                   '觑'  0.0036


======================================================================
  PROMPT_C — Logit Lens at Layer 25
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' female'  0.6118
       2.               ' male'  0.0615
       3.              'Female'  0.0215
       4.                   '女'  0.0195
       5.             ' Female'  0.0137
       6.                   '⼥'  0.0103
       7.                 ' **'  0.0100
       8.                  '女性'  0.0085
       9.              'female'  0.0072
      10.             'osexual'  0.0063
      11.                 'cis'  0.0044
      12.               ' Male'  0.0039
      13.             '-scenes'  0.0032
      14.        ' undisclosed'  0.0030
      15.                  '不限'  0.0027
      16.            ' females'  0.0022
      17.              ' woman'  0.0020
      18.                '(":/'  0.0019
      19.                 'yny'  0.0017
      20.              ' women'  0.0017


======================================================================
  PROMPT_C — Logit Lens at Layer 26
======================================================================

  --- var1 ---
  Position   77token='p'
       1.                 ' **'  0.7363
       2.             ' female'  0.1221
       3.             ' Female'  0.0494
       4.                  '**'  0.0196
       5.                   '女'  0.0134
       6.              'Female'  0.0091
       7.               ' Male'  0.0018
       8.                 ' \n'  0.0014
       9.                  '女性'  0.0013
      10.                  ' *'  0.0013
      11.               ' male'  0.0011
      12.        ' undisclosed'  0.0009
      13.              'female'  0.0007
      14.                   '⼥'  0.0007
      15.            ' females'  0.0006
      16.                '  \n'  0.0005
      17.                  ' C'  0.0004
      18.                 '\\_'  0.0004
      19.              ' White'  0.0004
      20.        ' unspecified'  0.0004


======================================================================
  PROMPT_C — Logit Lens at Layer 27
======================================================================

  --- var1 ---
  Position   77token='p'
       1.             ' Female'  0.9972
       2.               ' Male'  0.0017
       3.              'Female'  0.0007
       4.                 ' **'  0.0003
       5.             ' female'  0.0000
       6.                ' Fem'  0.0000
       7.                   '女'  0.0000
       8.              ' Woman'  0.0000
       9.                 ' \n'  0.0000
      10.                ' Non'  0.0000
      11.             ' Gender'  0.0000
      12.                'Male'  0.0000
      13.              ' Women'  0.0000
      14.             '_female'  0.0000
      15.              ' Femin'  0.0000
      16.              'female'  0.0000
      17.          ' Caucasian'  0.0000
      18.                  '女性'  0.0000
      19.                '  \n'  0.0000
      20.                   '⼥'  0.0000



