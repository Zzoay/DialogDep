

rel_dct = {
    'root': '根节点',
    'sasubj-obj': '同主同宾',
    'sasubj': '同主语',
    'dfsubj': '不同主语',
    'subj': '主语',
    'subj-in': '内部主语',
    'obj': '宾语',
    'pred': '谓语',
    'att': '定语',
    'adv': '状语',
    'cmp': '补语',
    'coo': '并列',
    'pobj': '介宾',
    'iobj': '间宾',
    'de': '的',
    'adjct': '附加',
    'app': '称呼',
    'exp': '解释',
    'punc': '标点',
    'frag': '片段',
    'repet': '重复',
    # rst
    'attr': '归属',
    'bckg': '背景',
    'cause': '因果',
    'comp': '比较',
    'cond': '状况',
    'cont': '对比',
    'elbr': '阐述',
    'enbm': '目的',
    'eval': '评价',
    'expl': '解释-例证',
    'joint': '联合',
    'manner': '方式',
    'rstm': '重申',
    'temp': '时序',
    'tp-chg': '主题变更',
    'prob-sol': '问题-解决',
    'qst-ans': '疑问-回答',
    'stm-rsp': '陈述-回应',
    'req-proc': '需求-处理',
}

rel2id = {}
for i, (key, value) in enumerate(rel_dct.items()):
    rel2id[key] = i


punct_lst = ['，', '.', '。', '!', '?', '~', '...', '......', ',', ':', '：', ';']


weak_signals = [
    ['说', '表示', '看到', '显示', '知道', '认为', '希望', '指出'],
    ['如果', '假如', '的话', '若', '如'],
    ['因为', '所以', '导致', '因此', '造成', '由于', '因而'],
    ['但是', '可是', '但', '竟', '却', '不过', '居然', '而是'],
    ['以及', '也', '并', '并且', '又', '或者'],
    ['对于', '自从', '上次'],
    ['明天', '晚上', '到时候', '再', '然后', '接下来', '最后', '随后'],
    ['为了', '使', '为 的', '为 了'],
    ['通过', '必须', '点击'],
    ['对 吗', '是 吗', '对 吧', '是 吧', '对 ?'],
    ['更', '比', '只'],
    ['解释', '比如', '例如', '是 这样'],
    ['理想', '真 棒', '太 棒', '真差', '太 差', '不 行', '扯皮', '这么 麻烦'],
]

weak_labels = [
    rel2id['attr'],
    rel2id['cond'],
    rel2id['cause'],
    rel2id['cont'],
    rel2id['joint'],
    rel2id['bckg'],
    rel2id['temp'],
    rel2id['enbm'],
    rel2id['manner'],
    rel2id['rstm'],
    rel2id['comp'],
    rel2id['expl'],
    rel2id['eval'],
]