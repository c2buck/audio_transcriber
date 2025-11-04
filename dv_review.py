"""
Domestic Violence Word List Analyzer
Analyzes transcriptions for indicators of domestic violence, coercive control, abuse, threats,
evidence tampering, police investigation references, and related concerns.

Features:
- Comprehensive Australian word list and slang
- Ranked scoring system
- Per-recording analysis
- Top 10 priority review list
"""

import re
from typing import Dict, List, Tuple, Any
from pathlib import Path
from collections import defaultdict


class DVWordListAnalyzer:
    """Analyzer for domestic violence indicators in transcriptions."""
    
    def __init__(self):
        """Initialize the analyzer with word lists and scoring weights."""
        self.word_categories = self._build_word_categories()
        self.category_weights = self._build_category_weights()
        
    def _build_word_categories(self) -> Dict[str, List[str]]:
        """Build comprehensive word lists for different DV-related categories."""
        return {
            # Direct threats and violence
            'threats': [
                'kill', 'murder', 'die', 'dead', 'hurt', 'harm', 'bash', 'beat', 'punch', 'hit',
                'smash', 'destroy', 'ruin', 'end you', 'finish you', 'get you', 'sort you out',
                'deal with you', 'settle this', 'revenge', 'payback', 'you\'ll pay', 'you\'ll regret',
                'watch your back', 'you\'re dead', 'game over', 'no more', 'last time', 'this is it',
                'threaten', 'menace', 'intimidate', 'scare', 'frighten', 'terrify', 'warning',
                'better watch out', 'you better', 'or else', 'consequences', 'you\'ll see',
                # Threats involving children/pregnancy
                'hurt the baby', 'kill the baby', 'harm the baby', 'hurt your baby', 'kill your baby',
                'harm your baby', 'hurt the child', 'kill the child', 'harm the child', 'hurt your child',
                'kill your child', 'harm your child', 'hurt the kids', 'kill the kids', 'harm the kids',
                'hurt your kids', 'kill your kids', 'harm your kids', 'hurt children', 'kill children',
                'harm children', 'take the kids', 'take your kids', 'take the children', 'take your children',
                'lose the kids', 'lose your kids', 'lose the children', 'lose your children', 'lose custody',
                'never see the kids', 'never see your kids', 'never see the children', 'never see your children',
                'get rid of the baby', 'get rid of it', 'get rid of the kid', 'get rid of the kids',
                # Australian slang
                'do you in', 'knock you', 'flatten you', 'lay you out', 'put you down',
                'take you out', 'get rid of you', 'dispose of you', 'make you disappear'
            ],
            
            # Physical abuse
            'physical_abuse': [
                'hit', 'hitting', 'strike', 'striking', 'slap', 'slapping', 'punch', 'punching',
                'kick', 'kicking', 'push', 'pushing', 'shove', 'shoving', 'grab', 'grabbing',
                'choke', 'choking', 'strangle', 'strangling', 'restrain', 'restraining',
                'force', 'forcing', 'throw', 'throwing', 'drag', 'dragging', 'pull', 'pulling',
                'pin', 'pinning', 'hold down', 'held down', 'trap', 'trapping', 'corner',
                'cornering', 'block', 'blocking', 'prevent', 'preventing', 'stop me', 'can\'t leave',
                'won\'t let', 'not allowed', 'forbidden', 'restricted', 'locked', 'locked in',
                'trapped', 'stuck', 'can\'t escape', 'can\'t get away', 'no way out',
                # Physical harm involving pregnancy/children
                'hit the baby', 'hit your baby', 'punch the baby', 'punch your baby', 'kick the baby',
                'kick your baby', 'hurt the baby', 'hurt your baby', 'hit the child', 'hit your child',
                'punch the child', 'punch your child', 'kick the child', 'kick your child', 'hurt the child',
                'hurt your child', 'hit the kids', 'hit your kids', 'punch the kids', 'punch your kids',
                'kick the kids', 'kick your kids', 'hurt the kids', 'hurt your kids', 'beat the kids',
                'beat your kids', 'beat the child', 'beat your child', 'bash the kids', 'bash your kids',
                'hit while pregnant', 'punch while pregnant', 'kick while pregnant', 'hurt while pregnant',
                'hit when pregnant', 'punch when pregnant', 'kick when pregnant', 'hurt when pregnant',
                'harm the pregnancy', 'hurt the pregnancy', 'damage the pregnancy', 'harm the unborn',
                'hurt the unborn', 'damage the unborn', 'hurt the fetus', 'harm the fetus',
                # Australian slang
                'king hit', 'sucker punch', 'coward punch', 'one punch', 'glass', 'bottle',
                'bash', 'bashing', 'flog', 'flogging', 'belt', 'belting', 'whack', 'whacking',
                'clobber', 'clobbering', 'thump', 'thumping', 'wallop', 'walloping'
            ],
            
            # Coercive control
            'coercive_control': [
                'control', 'controlling', 'must', 'have to', 'must do', 'required', 'demand',
                'demanding', 'insist', 'insisting', 'force', 'forcing', 'make you', 'made you',
                'won\'t let', 'not allowed', 'forbidden', 'can\'t', 'not permitted', 'prohibited',
                'restricted', 'limited', 'isolate', 'isolating', 'cut off', 'cut you off',
                'separate', 'separating', 'divide', 'dividing', 'alienate', 'alienating',
                'monitor', 'monitoring', 'track', 'tracking', 'watch', 'watching', 'check',
                'checking', 'surveillance', 'spy', 'spying', 'follow', 'following', 'stalk',
                'stalking', 'where are you', 'where were you', 'who were you with', 'why were you',
                'what did you do', 'explain', 'account for', 'answer for', 'justify',
                'no friends', 'can\'t see', 'not allowed to see', 'stay away from', 'avoid',
                'don\'t talk to', 'can\'t talk to', 'forbidden to speak', 'not allowed to contact',
                # Financial control
                'money', 'bank account', 'credit card', 'no access', 'can\'t have', 'can\'t spend',
                'control money', 'your money', 'my money', 'our money', 'spend', 'spending',
                'allowance', 'budget', 'waste', 'wasting', 'squander', 'squandering',
                # Australian slang
                'dole', 'centrelink', 'welfare', 'pension', 'benefits', 'no cash', 'broke',
                'no money', 'skint', 'penniless'
            ],
            
            # Emotional/psychological abuse
            'emotional_abuse': [
                'stupid', 'idiot', 'moron', 'fool', 'dumb', 'useless', 'worthless', 'pathetic',
                'weak', 'pathetic', 'loser', 'failure', 'no good', 'good for nothing',
                'nobody', 'nothing', 'nobody wants you', 'no one likes you', 'everyone hates you',
                'you\'re nothing', 'you\'re worthless', 'you don\'t matter', 'you\'re replaceable',
                'insult', 'insulting', 'belittle', 'belittling', 'humiliate', 'humiliating',
                'embarrass', 'embarrassing', 'shame', 'shaming', 'degrade', 'degrading',
                'blame', 'blaming', 'fault', 'your fault', 'always your fault', 'everything is your fault',
                'guilt', 'guilty', 'make you feel guilty', 'guilt trip', 'manipulate', 'manipulating',
                'gaslight', 'gaslighting', 'that never happened', 'you\'re imagining', 'you\'re crazy',
                'you\'re mental', 'you\'re insane', 'you\'re delusional', 'not real', 'didn\'t happen',
                'you\'re wrong', 'you\'re mistaken', 'you don\'t remember', 'you misremembered',
                'silent treatment', 'ignore', 'ignoring', 'stonewall', 'stonewalling', 'shut down',
                'withdraw', 'withdrawing', 'cold shoulder', 'punish', 'punishing', 'punishment',
                # Australian slang
                'drongo', 'galah', 'dingo', 'dropkick', 'spud', 'spastic', 'retard', 'mongrel',
                'bastard', 'arsehole', 'dickhead', 'shithead', 'wanker', 'dick', 'prick'
            ],
            
            # Evidence tampering
            'evidence_tampering': [
                'delete', 'deleting', 'erase', 'erasing', 'remove', 'removing', 'destroy',
                'destroying', 'get rid of', 'throw away', 'discard', 'discarding', 'eliminate',
                'eliminating', 'wipe', 'wiping', 'clear', 'clearing', 'clean', 'cleaning',
                'remove evidence', 'delete evidence', 'destroy evidence', 'get rid of evidence',
                'no evidence', 'no proof', 'can\'t prove', 'nothing to prove', 'no one will believe',
                'no witnesses', 'no one saw', 'no one knows', 'nobody knows', 'secret', 'keep secret',
                'don\'t tell', 'can\'t tell', 'mustn\'t tell', 'forbidden to tell', 'not allowed to say',
                'cover up', 'covering up', 'hide', 'hiding', 'conceal', 'concealing', 'suppress',
                'suppressing', 'bury', 'burying', 'sweep under', 'under the rug', 'forget',
                'make forget', 'pretend', 'pretending', 'act like', 'play dumb', 'act innocent',
                'records', 'recordings', 'audio', 'video', 'photos', 'pictures', 'messages',
                'texts', 'emails', 'calls', 'call log', 'phone records', 'delete messages',
                'clear messages', 'remove messages', 'delete photos', 'remove photos',
                'delete recordings', 'remove recordings', 'delete video', 'remove video'
            ],
            
            # Police investigation references
            'police_investigation': [
                'police', 'copper', 'cop', 'cops', 'officer', 'officers', 'detective', 'detectives',
                'investigation', 'investigating', 'investigate', 'inquiry', 'enquiring', 'enquire',
                'statement', 'statements', 'give statement', 'make statement', 'written statement',
                'witness', 'witnesses', 'testify', 'testifying', 'testimony', 'court', 'courts',
                'charges', 'charged', 'arrest', 'arrested', 'arresting', 'warrant', 'warrants',
                'search', 'searching', 'search warrant', 'raid', 'raiding', 'seize', 'seizing',
                'seizure', 'evidence', 'proof', 'proof of', 'prove', 'proving', 'proven',
                'body worn', 'body worn camera', 'bwc', 'body camera', 'bodycam', 'dash cam',
                'dashcam', 'cctv', 'security camera', 'surveillance', 'surveillance footage',
                'recording', 'recordings', 'audio recording', 'video recording', 'phone recording',
                'call recording', 'recorded', 'taped', 'taping', 'wiretap', 'wiretapping',
                'subpoena', 'subpoenas', 'summons', 'witness statement', 'deposition',
                # Australian police terms
                'afp', 'australian federal police', 'state police', 'local police', 'station',
                'police station', 'precinct', 'detective branch', 'cib', 'criminal investigation',
                'domestic violence unit', 'dvu', 'family violence', 'avlo', 'apprehended violence order',
                'avo', 'intervention order', 'protection order', 'restraining order', 'dvo',
                'domestic violence order', 'breach', 'breaching', 'breached', 'violation',
                'violating', 'violated'
            ],
            
            # Withdrawal of complaints / non-cooperation
            'withdrawal_non_cooperation': [
                'withdraw', 'withdrawing', 'withdrawn', 'withdrawal', 'drop', 'dropping', 'dropped',
                'drop charges', 'withdraw charges', 'withdraw complaint', 'drop complaint',
                'cancel', 'cancelling', 'cancelled', 'retract', 'retracting', 'retracted',
                'recant', 'recanting', 'recanted', 'take back', 'taking back', 'took back',
                'change story', 'changing story', 'changed story', 'different story', 'new story',
                'not true', 'wasn\'t true', 'not real', 'didn\'t happen', 'never happened',
                'made up', 'lie', 'lied', 'lying', 'false', 'false statement', 'false complaint',
                'mistake', 'mistaken', 'wrong', 'wrong about', 'regret', 'regretting', 'sorry',
                'apologize', 'apologizing', 'apologised', 'forgive', 'forgiving', 'forgave',
                'forget', 'forgetting', 'forgot', 'move on', 'moving on', 'let it go', 'let\'s forget',
                'don\'t want', 'don\'t want to', 'don\'t want charges', 'don\'t want to press charges',
                'don\'t want to go to court', 'don\'t want police', 'don\'t want cops',
                'leave police out', 'keep police out', 'no police', 'not involving police',
                'don\'t involve police', 'can\'t involve police', 'mustn\'t involve police',
                'not cooperate', 'not cooperating', 'won\'t cooperate', 'refuse', 'refusing',
                'refused', 'refusal', 'won\'t help', 'can\'t help', 'not helping', 'no help',
                'don\'t help police', 'won\'t assist', 'not assisting', 'refuse to assist',
                'refuse to help', 'refuse to cooperate', 'won\'t testify', 'refuse to testify',
                'not testifying', 'won\'t give statement', 'refuse to give statement',
                'won\'t make statement', 'refuse to make statement', 'keep quiet', 'stay quiet',
                'say nothing', 'don\'t say anything', 'can\'t say', 'mustn\'t say', 'not allowed to say',
                # Australian slang
                'dob', 'dob in', 'dobbed', 'dobbin\'', 'snitch', 'snitching', 'snitched',
                'tell on', 'telling on', 'told on', 'grass', 'grassing', 'grassed', 'rat',
                'ratting', 'ratted', 'don\'t dob', 'don\'t snitch', 'don\'t tell', 'keep mouth shut'
            ],
            
            # Version change / story manipulation
            'version_change': [
                'change story', 'changing story', 'changed story', 'different story', 'new story',
                'tell different', 'tell differently', 'say different', 'say differently',
                'alternative version', 'different version', 'other version', 'another version',
                'tell police', 'tell cops', 'say to police', 'say to cops', 'what to say',
                'what to tell', 'how to say', 'how to tell', 'script', 'scripted', 'rehearse',
                'rehearsing', 'practice', 'practicing', 'practice story', 'rehearse story',
                'memorize', 'memorizing', 'remember to say', 'don\'t forget to say',
                'make sure you say', 'be sure to say', 'important to say', 'must say',
                'have to say', 'need to say', 'should say', 'better say', 'better tell',
                'tell them', 'say this', 'not that', 'don\'t say that', 'can\'t say that',
                'mustn\'t say that', 'wrong thing', 'right thing', 'correct thing', 'proper thing',
                'right way', 'wrong way', 'correct way', 'proper way', 'how it should be',
                'how it must be', 'how it has to be', 'how it needs to be', 'how it\'s supposed to be',
                'stick to', 'stick to the story', 'stick to script', 'follow script', 'follow story',
                'don\'t deviate', 'don\'t change', 'don\'t add', 'don\'t mention', 'don\'t bring up',
                'don\'t talk about', 'avoid', 'avoiding', 'skip', 'skipping', 'leave out', 'leaving out',
                'omit', 'omitting', 'forget', 'forgetting', 'don\'t remember', 'act like you don\'t remember',
                'pretend', 'pretending', 'act', 'acting', 'play', 'playing', 'fake', 'faking',
                'lie', 'lying', 'not true', 'untrue', 'false', 'fabricate', 'fabricating',
                'invent', 'inventing', 'make up', 'making up', 'made up', 'concoct', 'concocting',
                'story', 'stories', 'version', 'versions', 'account', 'accounts', 'narrative',
                'narratives', 'tale', 'tales', 'explanation', 'explanations', 'description',
                'descriptions', 'report', 'reports', 'statement', 'statements', 'testimony',
                'testimonies', 'evidence', 'proof', 'alibi', 'alibis', 'excuse', 'excuses',
                'cover story', 'cover-up', 'cover up', 'smoke screen', 'distraction', 'deflection'
            ],
            
            # Sexual abuse / coercion
            'sexual_abuse': [
                'rape', 'raping', 'raped', 'sexual assault', 'assaulted', 'assaulting',
                'force', 'forcing', 'forced', 'make you', 'made you', 'coerce', 'coercing',
                'coerced', 'pressure', 'pressuring', 'pressured', 'convince', 'convincing',
                'convinced', 'persuade', 'persuading', 'persuaded', 'talk into', 'talked into',
                'no choice', 'no option', 'must do', 'have to do', 'required to do',
                'owe', 'owing', 'owed', 'you owe me', 'owe me', 'debt', 'payback', 'repayment',
                'right', 'my right', 'entitled', 'entitlement', 'deserve', 'deserving', 'owed this',
                'deserve this', 'entitled to this', 'right to this', 'my right to',
                'duty', 'your duty', 'obligation', 'your obligation', 'responsibility', 'your responsibility',
                'sex', 'sexual', 'intercourse', 'oral', 'anal', 'penetration', 'penetrating',
                'touch', 'touching', 'touched', 'fondle', 'fondling', 'fondled', 'grope',
                'groping', 'groped', 'molest', 'molesting', 'molested', 'abuse', 'abusing',
                'abused', 'violate', 'violating', 'violated', 'violation', 'exploit', 'exploiting',
                'exploited', 'exploitation', 'use', 'using', 'used', 'object', 'objectify',
                'objectifying', 'objectified', 'property', 'my property', 'belongs to me',
                'mine', 'own', 'owning', 'owned', 'possess', 'possessing', 'possessed',
                'possession', 'control', 'controlling', 'controlled'
            ],
            
            # Isolation / social control
            'isolation': [
                'isolate', 'isolating', 'isolated', 'isolation', 'separate', 'separating',
                'separated', 'separation', 'cut off', 'cutting off', 'cut you off', 'cut them off',
                'divide', 'dividing', 'divided', 'division', 'alienate', 'alienating', 'alienated',
                'alienation', 'distance', 'distancing', 'distanced', 'remove', 'removing', 'removed',
                'removal', 'prevent', 'preventing', 'prevented', 'prevention', 'stop', 'stopping',
                'stopped', 'block', 'blocking', 'blocked', 'bar', 'barring', 'barred', 'ban',
                'banning', 'banned', 'prohibit', 'prohibiting', 'prohibited', 'prohibition',
                'forbid', 'forbidding', 'forbidden', 'not allowed', 'can\'t', 'mustn\'t',
                'forbidden to', 'not permitted', 'not permitted to', 'prohibited from',
                'no contact', 'can\'t contact', 'mustn\'t contact', 'not allowed to contact',
                'don\'t contact', 'stop contacting', 'stop talking to', 'don\'t talk to',
                'can\'t talk to', 'mustn\'t talk to', 'not allowed to talk', 'forbidden to talk',
                'no friends', 'can\'t have friends', 'not allowed friends', 'forbidden friends',
                'no family', 'can\'t see family', 'not allowed to see family', 'forbidden family',
                'stay away from', 'keep away from', 'avoid', 'avoiding', 'avoided', 'shun',
                'shunning', 'shunned', 'ignore', 'ignoring', 'ignored', 'cut out', 'cutting out',
                'locked in', 'locked out', 'trapped', 'stuck', 'no escape', 'nowhere to go',
                'nowhere to turn', 'no one to turn to', 'no support', 'no help', 'alone',
                'lonely', 'isolated', 'cut off from everyone', 'cut off from world', 'separated from',
                'divided from', 'alienated from', 'removed from', 'prevented from', 'stopped from',
                'blocked from', 'barred from', 'banned from', 'prohibited from', 'forbidden from'
            ],
            
            # Stalking / monitoring
            'stalking_monitoring': [
                'stalk', 'stalking', 'stalked', 'stalking', 'follow', 'following', 'followed',
                'track', 'tracking', 'tracked', 'trace', 'tracing', 'traced', 'monitor', 'monitoring',
                'monitored', 'surveillance', 'watch', 'watching', 'watched', 'observe', 'observing',
                'observed', 'spy', 'spying', 'spied', 'check', 'checking', 'checked', 'keep tabs',
                'keeping tabs', 'kept tabs', 'track down', 'tracking down', 'tracked down',
                'find', 'finding', 'found', 'locate', 'locating', 'located', 'pinpoint', 'pinpointing',
                'pinpointed', 'hunt', 'hunting', 'hunted', 'pursue', 'pursuing', 'pursued',
                'chase', 'chasing', 'chased', 'trail', 'trailing', 'trailed', 'shadow', 'shadowing',
                'shadowed', 'tail', 'tailing', 'tailed', 'where are you', 'where were you',
                'where did you go', 'where are you going', 'where have you been', 'what are you doing',
                'what did you do', 'who were you with', 'who are you with', 'why were you there',
                'why did you go', 'explain', 'explaining', 'explained', 'account for', 'accounting for',
                'accounted for', 'answer for', 'answering for', 'answered for', 'justify', 'justifying',
                'justified', 'explain yourself', 'account for yourself', 'answer for yourself',
                'phone tracking', 'gps tracking', 'location tracking', 'phone location', 'find my phone',
                'where is your phone', 'check phone', 'checking phone', 'checked phone', 'look at phone',
                'looking at phone', 'looked at phone', 'read messages', 'reading messages', 'read texts',
                'reading texts', 'check messages', 'checking messages', 'checked messages', 'check texts',
                'checking texts', 'checked texts', 'go through phone', 'going through phone',
                'went through phone', 'search phone', 'searching phone', 'searched phone',
                'social media', 'facebook', 'instagram', 'twitter', 'snapchat', 'whatsapp',
                'check social media', 'checking social media', 'checked social media',
                'who are you talking to', 'who did you talk to', 'who are you messaging',
                'who did you message', 'who are you texting', 'who did you text'
            ],
            
            # Pregnancy and child harm
            'pregnancy_child_harm': [
                # Pregnancy terms
                'pregnant', 'pregnancy', 'unborn', 'fetus', 'foetus', 'baby', 'babies', 'expecting',
                'expectant', 'carrying', 'gestation', 'gestational', 'maternity', 'maternal', 'prenatal',
                'antenatal', 'obstetric', 'obstetrics', 'midwife', 'midwifery', 'ultrasound', 'scan',
                # Child terms
                'child', 'children', 'kid', 'kids', 'infant', 'infants', 'toddler', 'toddlers',
                'newborn', 'newborns', 'baby', 'babies', 'minor', 'minors', 'offspring', 'son', 'sons',
                'daughter', 'daughters', 'boy', 'boys', 'girl', 'girls', 'custody', 'custodial',
                # Threats to pregnancy
                'hurt the pregnancy', 'harm the pregnancy', 'damage the pregnancy', 'end the pregnancy',
                'terminate the pregnancy', 'lose the pregnancy', 'kill the pregnancy', 'get rid of the pregnancy',
                'hurt the unborn', 'harm the unborn', 'damage the unborn', 'kill the unborn', 'hurt the fetus',
                'harm the fetus', 'damage the fetus', 'kill the fetus', 'hurt the baby', 'harm the baby',
                'kill the baby', 'lose the baby', 'miscarry', 'miscarriage', 'miscarrying', 'cause miscarriage',
                'make you miscarry', 'force miscarriage', 'abort', 'abortion', 'aborting', 'force abortion',
                'make you abort', 'terminate', 'termination', 'terminating', 'force termination',
                'hurt while pregnant', 'harm while pregnant', 'hit while pregnant', 'punch while pregnant',
                'kick while pregnant', 'beat while pregnant', 'hurt when pregnant', 'harm when pregnant',
                'hit when pregnant', 'punch when pregnant', 'kick when pregnant', 'beat when pregnant',
                'you\'ll lose it', 'you\'ll lose the baby', 'you\'ll lose the pregnancy', 'you\'ll miscarry',
                'you\'ll have a miscarriage', 'you won\'t have it', 'you won\'t have the baby',
                'you won\'t have this baby', 'not having it', 'not having the baby', 'not keeping it',
                'not keeping the baby', 'get rid of it', 'get rid of the baby', 'get rid of the pregnancy',
                # Threats to children
                'hurt the child', 'harm the child', 'kill the child', 'hurt your child', 'harm your child',
                'kill your child', 'hurt the children', 'harm the children', 'kill the children',
                'hurt your children', 'harm your children', 'kill your children', 'hurt the kids',
                'harm the kids', 'kill the kids', 'hurt your kids', 'harm your kids', 'kill your kids',
                'take the kids', 'take your kids', 'take the children', 'take your children',
                'take them away', 'take them from you', 'lose the kids', 'lose your kids',
                'lose the children', 'lose your children', 'lose custody', 'lose custody of',
                'never see them', 'never see the kids', 'never see your kids', 'never see the children',
                'never see your children', 'you\'ll never see', 'you won\'t see', 'you can\'t see',
                'not allowed to see', 'forbidden to see', 'can\'t visit', 'not allowed to visit',
                'forbidden to visit', 'can\'t see them', 'can\'t see the kids', 'can\'t see your kids',
                'can\'t see the children', 'can\'t see your children', 'no contact', 'no contact with',
                'no contact with kids', 'no contact with children', 'no access', 'no access to',
                'no access to kids', 'no access to children', 'supervised visits', 'supervised contact',
                'restricted access', 'limited access', 'court order', 'restraining order against',
                # Physical harm to children
                'hit the child', 'hit your child', 'punch the child', 'punch your child',
                'kick the child', 'kick your child', 'beat the child', 'beat your child',
                'bash the child', 'bash your child', 'hit the kids', 'hit your kids',
                'punch the kids', 'punch your kids', 'kick the kids', 'kick your kids',
                'beat the kids', 'beat your kids', 'bash the kids', 'bash your kids',
                'hit the children', 'hit your children', 'punch the children', 'punch your children',
                'kick the children', 'kick your children', 'beat the children', 'beat your children',
                'bash the children', 'bash your children', 'hit the baby', 'hit your baby',
                'punch the baby', 'punch your baby', 'kick the baby', 'kick your baby',
                'beat the baby', 'beat your baby', 'bash the baby', 'bash your baby',
                # Coercive control using children
                'using the kids', 'using the children', 'using the child', 'using the baby',
                'through the kids', 'through the children', 'through the child', 'through the baby',
                'using custody', 'threaten custody', 'threatening custody', 'custody battle',
                'custody fight', 'fight for custody', 'take custody', 'get custody', 'win custody',
                'lose custody', 'full custody', 'sole custody', 'shared custody', 'joint custody',
                'visitation', 'visitation rights', 'access rights', 'parenting time', 'contact order',
                # Australian terms
                'dca', 'department of child safety', 'child safety', 'cps', 'child protection',
                'child protection services', 'dcs', 'facs', 'family and community services',
                'documents', 'dhs', 'department of human services', 'centrelink', 'child support',
                'child support agency', 'csa', 'maintenance', 'child maintenance'
            ]
        }
    
    def _build_category_weights(self) -> Dict[str, float]:
        """Build scoring weights for each category. Higher weights = more severe/threatening."""
        return {
            # Very high severity - Direct threats of harm, assault, criminal offenses
            'threats': 50.0,  # Direct threats of violence/harm
            'pregnancy_child_harm': 50.0,  # Threats/harm to pregnancy/children (most severe)
            'physical_abuse': 40.0,  # Actual physical violence/assault
            'sexual_abuse': 45.0,  # Sexual assault/coercion (very severe)
            
            # High severity - Criminal activities and evidence manipulation
            'evidence_tampering': 30.0,  # Evidence tampering (criminal offense)
            'version_change': 25.0,  # Story manipulation/obstruction of justice
            
            # Medium-high severity - Coercive behaviors
            'withdrawal_non_cooperation': 20.0,  # Withdrawing complaints/obstruction
            'coercive_control': 18.0,  # Coercive control behaviors
            
            # Medium severity - Monitoring and control
            'stalking_monitoring': 15.0,  # Stalking/surveillance
            'isolation': 12.0,  # Isolation/social control
            
            # Lower severity but still concerning
            'police_investigation': 8.0,  # References to police (may be contextual)
            'emotional_abuse': 10.0  # Emotional/psychological abuse
        }
    
    def analyze_transcription(self, transcription_text: str, filename: str) -> Dict[str, Any]:
        """
        Analyze a single transcription for DV indicators.
        
        Args:
            transcription_text: The full transcription text
            filename: Name of the audio file
            
        Returns:
            Dictionary with analysis results including score, matches, and details
        """
        if not transcription_text or not transcription_text.strip():
            return {
                'filename': filename,
                'total_score': 0.0,
                'match_count': 0,
                'category_scores': {},
                'matches': {},
                'top_matches': []
            }
        
        # Normalize text for matching
        text_lower = transcription_text.lower()
        
        # Find matches in each category
        category_scores = {}
        matches = {}
        total_score = 0.0
        match_count = 0
        
        for category, words in self.word_categories.items():
            category_matches = []
            category_score = 0.0
            
            for word in words:
                # Use word boundaries for exact word matching
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                found_matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in found_matches:
                    # Get context around the match
                    start = max(0, match.start() - 30)
                    end = min(len(transcription_text), match.end() + 30)
                    context = transcription_text[start:end].strip()
                    
                    category_matches.append({
                        'word': word,
                        'position': match.start(),
                        'context': context
                    })
                    
                    # Add score based on category weight
                    weight = self.category_weights.get(category, 1.0)
                    category_score += weight
                    match_count += 1
            
            if category_matches:
                category_scores[category] = category_score
                matches[category] = category_matches
                total_score += category_score
        
        # Get top matches (sorted by category weight)
        top_matches = []
        for category, category_matches in matches.items():
            weight = self.category_weights.get(category, 1.0)
            for match in category_matches:
                top_matches.append({
                    'category': category,
                    'word': match['word'],
                    'context': match['context'],
                    'weight': weight
                })
        
        # Sort by weight (descending)
        top_matches.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'filename': filename,
            'total_score': round(total_score, 2),
            'match_count': match_count,
            'category_scores': category_scores,
            'matches': matches,
            'top_matches': top_matches[:20]  # Top 20 matches
        }
    
    def analyze_batch(self, transcription_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple transcriptions.
        
        Args:
            transcription_results: List of transcription result dictionaries
            
        Returns:
            Dictionary with batch analysis results including top 10 recordings
        """
        analyses = []
        
        for result in transcription_results:
            if not result.get('success', False):
                continue
            
            filename = Path(result.get('file_path', '')).name
            transcription = result.get('transcription', '')
            
            # Combine all segments if available
            if not transcription and 'segments' in result:
                segments = result.get('segments', [])
                transcription = ' '.join([seg.get('text', '') for seg in segments])
            
            analysis = self.analyze_transcription(transcription, filename)
            analysis['file_path'] = result.get('file_path', '')
            analysis['duration'] = result.get('duration', 0.0)
            analyses.append(analysis)
        
        # Sort by score (descending)
        analyses.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Get top 10
        top_10 = analyses[:10] if len(analyses) >= 10 else analyses
        
        return {
            'analyses': analyses,
            'top_10': top_10,
            'total_recordings': len(analyses),
            'recordings_with_matches': len([a for a in analyses if a['match_count'] > 0])
        }
    
    def get_category_names(self) -> Dict[str, str]:
        """Get human-readable category names."""
        return {
            'threats': 'Threats and Violence',
            'physical_abuse': 'Physical Abuse',
            'sexual_abuse': 'Sexual Abuse/Coercion',
            'coercive_control': 'Coercive Control',
            'emotional_abuse': 'Emotional/Psychological Abuse',
            'evidence_tampering': 'Evidence Tampering',
            'police_investigation': 'Police Investigation References',
            'withdrawal_non_cooperation': 'Withdrawal/Non-Cooperation',
            'version_change': 'Version Change/Story Manipulation',
            'isolation': 'Isolation/Social Control',
            'stalking_monitoring': 'Stalking/Monitoring',
            'pregnancy_child_harm': 'Pregnancy/Child Harm'
        }

