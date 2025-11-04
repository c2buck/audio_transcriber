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
        """
        Build comprehensive word lists for different DV-related categories.
        Prioritizes phrases over single words to reduce false positives.
        """
        return {
            # Direct threats and violence - PHRASES FIRST (high priority)
            'threats': [
                # Critical threat phrases - highest priority
                'i\'ll kill you', 'i will kill you', 'going to kill you', 'gonna kill you',
                'i\'ll murder you', 'i will murder you', 'going to murder you',
                'you\'re dead', 'you are dead', 'you\'ll be dead', 'you will be dead',
                'i\'ll hurt you', 'i will hurt you', 'going to hurt you', 'gonna hurt you',
                'i\'ll get you', 'i will get you', 'going to get you', 'gonna get you',
                'i\'ll end you', 'i will end you', 'going to end you', 'finish you',
                'sort you out', 'deal with you', 'settle this', 'you\'ll pay', 'you will pay',
                'you\'ll regret', 'you will regret', 'watch your back', 'game over',
                'or else', 'you better', 'better watch out', 'consequences', 'you\'ll see',
                # Threats to children/pregnancy - phrases only
                'hurt the baby', 'kill the baby', 'harm the baby', 'hurt your baby', 'kill your baby',
                'harm your baby', 'hurt the child', 'kill the child', 'harm the child', 'hurt your child',
                'kill your child', 'harm your child', 'hurt the kids', 'kill the kids', 'harm the kids',
                'hurt your kids', 'kill your kids', 'harm your kids', 'take the kids', 'take your kids',
                'take the children', 'take your children', 'take them away', 'take them from you',
                'lose the kids', 'lose your kids', 'lose the children', 'lose your children', 'lose custody',
                'never see the kids', 'never see your kids', 'never see the children', 'never see your children',
                'never see them', 'you\'ll never see', 'you won\'t see', 'you can\'t see',
                'get rid of the baby', 'get rid of it', 'get rid of the kid', 'get rid of the kids',
                # Australian threat phrases
                'do you in', 'knock you out', 'flatten you', 'lay you out', 'put you down',
                'take you out', 'get rid of you', 'dispose of you', 'make you disappear',
                'bash you', 'flog you', 'belt you', 'thump you',
                # Only the most threatening single words (context-dependent but worth flagging)
                'kill', 'murder', 'die', 'dead', 'hurt', 'harm', 'bash', 'beat', 'punch', 'hit',
                'smash', 'destroy', 'ruin', 'revenge', 'payback'
            ],
            
            # Physical abuse - PHRASES FIRST
            'physical_abuse': [
                # Physical violence phrases - highest priority
                'hit you', 'hitting you', 'hit me', 'hitting me', 'hit her', 'hitting her',
                'punch you', 'punching you', 'punch me', 'punching me', 'punch her', 'punching her',
                'kick you', 'kicking you', 'kick me', 'kicking me', 'kick her', 'kicking her',
                'beat you', 'beating you', 'beat me', 'beating me', 'beat her', 'beating her',
                'bash you', 'bashing you', 'bash me', 'bashing me', 'bash her', 'bashing her',
                'slap you', 'slapping you', 'slap me', 'slapping me', 'slap her', 'slapping her',
                'choke you', 'choking you', 'choke me', 'choking me', 'choke her', 'choking her',
                'strangle you', 'strangling you', 'strangle me', 'strangling me', 'strangle her', 'strangling her',
                'grab you', 'grabbing you', 'grab me', 'grabbing me', 'grab her', 'grabbing her',
                'push you', 'pushing you', 'push me', 'pushing me', 'push her', 'pushing her',
                'shove you', 'shoving you', 'shove me', 'shoving me', 'shove her', 'shoving her',
                'throw you', 'throwing you', 'throw me', 'throwing me', 'throw her', 'throwing her',
                'drag you', 'dragging you', 'drag me', 'dragging me', 'drag her', 'dragging her',
                'force you', 'forcing you', 'force me', 'forcing me', 'force her', 'forcing her',
                'hold you down', 'held you down', 'hold me down', 'held me down', 'hold her down', 'held her down',
                'pin you down', 'pinned you down', 'pin me down', 'pinned me down', 'pin her down', 'pinned her down',
                'locked you in', 'locked me in', 'locked her in', 'trapped you', 'trapped me', 'trapped her',
                'can\'t leave', 'won\'t let you leave', 'won\'t let me leave', 'won\'t let her leave',
                'can\'t escape', 'can\'t get away', 'no way out', 'not allowed to leave',
                # Physical harm involving pregnancy/children - phrases only
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
                # Australian violent phrases
                'king hit', 'sucker punch', 'coward punch', 'one punch', 'glass you', 'bottle you',
                'flog you', 'belting you', 'whack you', 'clobber you', 'thump you', 'wallop you'
            ],
            
            # Coercive control - PHRASES FIRST
            'coercive_control': [
                # Control phrases - highest priority
                'make you', 'made you', 'forcing you', 'forced you', 'make me', 'made me',
                'won\'t let you', 'not allowed to', 'forbidden to', 'can\'t', 'must do',
                'have to do', 'required to', 'demand you', 'demanding you', 'insist you', 'insisting you',
                'cut you off', 'cut me off', 'cut her off', 'isolating you', 'isolated you',
                'prevent you from', 'preventing you from', 'stop you from', 'stopping you from',
                'block you from', 'blocking you from', 'not permitted to', 'prohibited from',
                'restricted from', 'limited to', 'no access to', 'can\'t have', 'can\'t spend',
                # Surveillance and monitoring phrases
                'where are you', 'where were you', 'who were you with', 'why were you',
                'what did you do', 'explain yourself', 'account for yourself', 'answer for yourself',
                'justify yourself', 'following you', 'stalking you', 'tracking you', 'monitoring you',
                'watching you', 'checking on you', 'spying on you', 'surveillance on you',
                # Social isolation phrases
                'no friends', 'can\'t see', 'not allowed to see', 'stay away from', 'avoid',
                'don\'t talk to', 'can\'t talk to', 'forbidden to speak', 'not allowed to contact',
                'no contact with', 'cut off from', 'separated from', 'alienated from',
                # Financial control phrases
                'control your money', 'control the money', 'your money', 'my money', 'our money',
                'can\'t spend', 'not allowed to spend', 'forbidden to spend', 'no access to money',
                'no cash', 'no money', 'broke', 'skint', 'penniless', 'spending allowance',
                'waste money', 'wasting money', 'squander money', 'squandering money'
            ],
            
            # Emotional/psychological abuse - PHRASES FIRST
            'emotional_abuse': [
                # Degrading phrases - highest priority
                'you\'re stupid', 'you are stupid', 'you\'re an idiot', 'you are an idiot',
                'you\'re useless', 'you are useless', 'you\'re worthless', 'you are worthless',
                'you\'re pathetic', 'you are pathetic', 'you\'re a loser', 'you are a loser',
                'you\'re a failure', 'you are a failure', 'no good', 'good for nothing',
                'nobody wants you', 'no one likes you', 'everyone hates you',
                'you\'re nothing', 'you are nothing', 'you\'re worthless', 'you are worthless',
                'you don\'t matter', 'you\'re replaceable', 'you are replaceable',
                # Blame and guilt phrases
                'your fault', 'always your fault', 'everything is your fault', 'all your fault',
                'make you feel guilty', 'guilt trip', 'guilt you', 'guilting you',
                # Gaslighting phrases
                'that never happened', 'you\'re imagining', 'you are imagining', 'you\'re crazy',
                'you are crazy', 'you\'re mental', 'you are mental', 'you\'re insane', 'you are insane',
                'you\'re delusional', 'you are delusional', 'not real', 'didn\'t happen',
                'you\'re wrong', 'you are wrong', 'you\'re mistaken', 'you are mistaken',
                'you don\'t remember', 'you misremembered', 'that\'s not what happened',
                # Manipulation and control phrases
                'silent treatment', 'ignoring you', 'stonewalling you', 'shut you down',
                'withdrawing from you', 'cold shoulder', 'punishing you', 'punishment for you',
                # Australian degrading phrases
                'you\'re a drongo', 'you\'re a galah', 'you\'re a dropkick', 'you\'re a mongrel',
                'you\'re a bastard', 'you\'re an arsehole', 'you\'re a dickhead', 'you\'re a shithead',
                'you\'re a wanker', 'you\'re a dick', 'you\'re a prick'
            ],
            
            # Evidence tampering - PHRASES FIRST
            'evidence_tampering': [
                # Evidence destruction phrases - highest priority
                'delete evidence', 'destroy evidence', 'remove evidence', 'get rid of evidence',
                'erase evidence', 'wipe evidence', 'clear evidence', 'eliminate evidence',
                'delete the evidence', 'destroy the evidence', 'remove the evidence',
                'delete that evidence', 'destroy that evidence', 'remove that evidence',
                # No evidence phrases
                'no evidence', 'no proof', 'can\'t prove', 'nothing to prove', 'no one will believe',
                'no witnesses', 'no one saw', 'no one knows', 'nobody knows',
                # Secrecy and cover-up phrases
                'keep secret', 'keep it secret', 'don\'t tell', 'can\'t tell', 'mustn\'t tell',
                'forbidden to tell', 'not allowed to say', 'cover up', 'covering up',
                'hide the evidence', 'hide evidence', 'conceal evidence', 'concealing evidence',
                'suppress evidence', 'suppressing evidence', 'bury the evidence', 'burying evidence',
                'sweep under the rug', 'under the rug', 'make forget', 'pretend it didn\'t happen',
                'act like nothing happened', 'play dumb', 'act innocent',
                # Digital evidence tampering phrases
                'delete messages', 'delete the messages', 'clear messages', 'clear the messages',
                'remove messages', 'remove the messages', 'delete photos', 'delete the photos',
                'remove photos', 'remove the photos', 'delete recordings', 'delete the recordings',
                'remove recordings', 'remove the recordings', 'delete video', 'delete the video',
                'remove video', 'remove the video', 'delete call log', 'clear call log',
                'delete phone records', 'clear phone records',
                # Contextual words (useful for indicating evidence discussions)
                'evidence', 'proof', 'witnesses', 'witness', 'recording', 'recordings'
            ],
            
            # Police investigation references - PHRASES AND CONTEXTUAL WORDS
            'police_investigation': [
                # Police investigation phrases - highest priority
                'police investigation', 'police are investigating', 'police investigating',
                'police inquiry', 'police enquiry', 'police are looking into',
                'give statement to police', 'make statement to police', 'written statement to police',
                'statement for police', 'police statement', 'give police statement',
                'police witness', 'witness for police', 'testify against', 'testify for police',
                'police charges', 'charged by police', 'police arrest', 'arrested by police',
                'police warrant', 'search warrant', 'police search', 'police raid',
                'police evidence', 'evidence for police', 'police proof', 'prove to police',
                # Body worn camera and recording phrases
                'body worn camera', 'body worn', 'body camera', 'bodycam', 'bwc',
                'police recording', 'police audio', 'police video', 'police footage',
                'recorded by police', 'police wiretap', 'police surveillance',
                # Court and legal phrases
                'police court', 'court case', 'court order', 'police subpoena',
                'witness statement for police', 'police deposition',
                # Statement-related phrases
                'give statement', 'make statement', 'written statement', 'formal statement',
                'police statement', 'court statement', 'witness statement', 'official statement',
                'statement about', 'statement regarding', 'statement concerning',
                # Lawyer/legal representation phrases
                'talk to lawyer', 'speak to lawyer', 'meet with lawyer', 'see lawyer',
                'contact lawyer', 'call lawyer', 'lawyer said', 'lawyer told',
                'my lawyer', 'your lawyer', 'get a lawyer', 'hire lawyer',
                'legal representation', 'legal counsel', 'attorney', 'solicitor',
                # Australian police terms - phrases only
                'domestic violence unit', 'dvu', 'family violence unit',
                'apprehended violence order', 'avo', 'intervention order',
                'protection order', 'restraining order', 'domestic violence order', 'dvo',
                'breach of avo', 'breaching avo', 'breached avo', 'violation of order',
                # Threats about police
                'don\'t call police', 'can\'t call police', 'mustn\'t call police',
                'not allowed to call police', 'forbidden to call police',
                'no police', 'keep police out', 'leave police out', 'don\'t involve police',
                # Contextual single words (lower priority but useful indicators)
                'statement', 'statements', 'police', 'lawyer', 'lawyers', 'attorney',
                'solicitor', 'court', 'courts', 'witness', 'testimony', 'charges',
                'arrest', 'warrant', 'evidence', 'investigation', 'detective', 'officer'
            ],
            
            # Withdrawal of complaints / non-cooperation - PHRASES ONLY
            'withdrawal_non_cooperation': [
                # Withdrawal phrases - highest priority
                'drop charges', 'withdraw charges', 'withdraw complaint', 'drop complaint',
                'withdraw the charges', 'drop the charges', 'withdraw the complaint', 'drop the complaint',
                'take back the complaint', 'taking back the complaint', 'took back the complaint',
                'cancel the charges', 'cancelling the charges', 'cancelled the charges',
                'retract the complaint', 'retracting the complaint', 'retracted the complaint',
                'recant the statement', 'recanting the statement', 'recanted the statement',
                # Story change phrases
                'change story', 'changing story', 'changed story', 'different story', 'new story',
                'change the story', 'changing the story', 'changed the story',
                'tell different story', 'tell differently', 'say different story', 'say differently',
                'not true', 'wasn\'t true', 'not real', 'didn\'t happen', 'never happened',
                'made up', 'false statement', 'false complaint', 'wrong about',
                # Non-cooperation phrases
                'don\'t want charges', 'don\'t want to press charges', 'don\'t want to go to court',
                'don\'t want police', 'don\'t want cops', 'leave police out', 'keep police out',
                'no police', 'not involving police', 'don\'t involve police', 'can\'t involve police',
                'mustn\'t involve police', 'not cooperate', 'not cooperating', 'won\'t cooperate',
                'refuse to cooperate', 'won\'t help police', 'can\'t help police', 'don\'t help police',
                'won\'t assist police', 'not assisting police', 'refuse to assist', 'refuse to help',
                'refuse to cooperate', 'won\'t testify', 'refuse to testify', 'not testifying',
                'won\'t give statement', 'refuse to give statement', 'won\'t make statement',
                'refuse to make statement', 'keep quiet', 'stay quiet', 'say nothing',
                'don\'t say anything', 'can\'t say', 'mustn\'t say', 'not allowed to say',
                # Australian slang phrases
                'don\'t dob', 'don\'t dob in', 'don\'t snitch', 'don\'t tell', 'keep mouth shut',
                'don\'t tell on', 'don\'t grass', 'don\'t rat'
            ],
            
            # Version change / story manipulation - PHRASES ONLY
            'version_change': [
                # Story manipulation phrases - highest priority
                'change story', 'changing story', 'changed story', 'different story', 'new story',
                'change the story', 'changing the story', 'changed the story',
                'tell different story', 'tell differently', 'say different story', 'say differently',
                'alternative version', 'different version', 'other version', 'another version',
                'tell police', 'tell cops', 'say to police', 'say to cops',
                'what to say', 'what to tell police', 'what to tell cops', 'how to say',
                'how to tell', 'how to tell police', 'how to say to police',
                'practice story', 'rehearse story', 'practice the story', 'rehearse the story',
                'remember to say', 'don\'t forget to say', 'make sure you say', 'be sure to say',
                'important to say', 'must say', 'have to say', 'need to say', 'should say',
                'better say', 'better tell', 'tell them', 'say this', 'not that',
                'don\'t say that', 'can\'t say that', 'mustn\'t say that',
                'wrong thing to say', 'right thing to say', 'correct thing to say', 'proper thing to say',
                'right way to say', 'wrong way to say', 'correct way to say', 'proper way to say',
                'how it should be', 'how it must be', 'how it has to be', 'how it needs to be',
                'how it\'s supposed to be', 'stick to the story', 'stick to script', 'follow script',
                'follow story', 'don\'t deviate', 'don\'t change the story', 'don\'t add to the story',
                'don\'t mention', 'don\'t bring up', 'don\'t talk about', 'avoid saying', 'skip that',
                'leave out', 'leaving out', 'act like you don\'t remember', 'pretend you don\'t remember',
                'pretend it didn\'t happen', 'act like nothing happened', 'play dumb', 'act innocent',
                'not true', 'untrue', 'false statement', 'fabricate story', 'fabricating story',
                'invent story', 'inventing story', 'make up story', 'making up story', 'concoct story',
                'cover story', 'cover-up', 'cover up', 'smoke screen'
            ],
            
            # Sexual abuse / coercion - PHRASES FIRST
            'sexual_abuse': [
                # Sexual assault phrases - highest priority
                'sexual assault', 'sexually assaulted', 'sexually assaulting', 'force you', 'forced you',
                'make you', 'made you', 'coerce you', 'coercing you', 'coerced you',
                'pressure you', 'pressuring you', 'pressured you', 'force me', 'forced me',
                'make me', 'made me', 'no choice', 'no option', 'must do', 'have to do',
                'required to do', 'you owe me', 'owe me', 'you owe', 'debt', 'payback',
                'my right', 'entitled to', 'deserve this', 'entitled to this', 'right to this',
                'my right to', 'your duty', 'your obligation', 'your responsibility',
                # Sexual coercion phrases
                'force sex', 'forced sex', 'forcing sex', 'make you have sex', 'made you have sex',
                'coerce into sex', 'coercing into sex', 'pressured into sex', 'pressure into sex',
                'no choice but to', 'no option but to', 'must have sex', 'have to have sex',
                'required to have sex', 'owe me sex', 'you owe me sex', 'my right to sex',
                'entitled to sex', 'deserve sex', 'your duty to have sex', 'your obligation',
                # Physical sexual abuse phrases
                'touch you', 'touching you', 'touched you', 'fondle you', 'fondling you',
                'fondled you', 'grope you', 'groping you', 'groped you', 'molest you',
                'molesting you', 'molested you', 'abuse you', 'abusing you', 'abused you',
                'violate you', 'violating you', 'violated you', 'exploit you', 'exploiting you',
                'exploited you', 'use you', 'using you', 'used you',
                # Objectification phrases
                'my property', 'belongs to me', 'mine', 'own you', 'owning you', 'owned you',
                'possess you', 'possessing you', 'possessed you', 'control you', 'controlling you',
                'controlled you', 'objectify you', 'objectifying you', 'objectified you',
                # Only most severe single words (context-dependent)
                'rape', 'raping', 'raped'
            ],
            
            # Isolation / social control - PHRASES ONLY
            'isolation': [
                # Isolation phrases - highest priority
                'cut you off', 'cut them off', 'cut me off', 'isolating you', 'isolated you',
                'separated you from', 'divided you from', 'alienated you from', 'removed you from',
                'prevent you from', 'preventing you from', 'stop you from', 'stopping you from',
                'block you from', 'blocking you from', 'bar you from', 'barring you from',
                'ban you from', 'banning you from', 'prohibit you from', 'prohibiting you from',
                'forbid you from', 'forbidding you from', 'not allowed to', 'forbidden to',
                'not permitted to', 'prohibited from', 'no contact', 'can\'t contact',
                'mustn\'t contact', 'not allowed to contact', 'don\'t contact', 'stop contacting',
                'stop talking to', 'don\'t talk to', 'can\'t talk to', 'mustn\'t talk to',
                'not allowed to talk', 'forbidden to talk', 'no friends', 'can\'t have friends',
                'not allowed friends', 'forbidden friends', 'no family', 'can\'t see family',
                'not allowed to see family', 'forbidden family', 'stay away from', 'keep away from',
                'cut out', 'cutting out', 'locked in', 'locked out', 'trapped', 'stuck',
                'no escape', 'nowhere to go', 'nowhere to turn', 'no one to turn to',
                'no support', 'no help', 'cut off from everyone', 'cut off from world',
                'separated from', 'divided from', 'alienated from', 'removed from',
                'prevented from', 'stopped from', 'blocked from', 'barred from',
                'banned from', 'prohibited from', 'forbidden from'
            ],
            
            # Stalking / monitoring - PHRASES ONLY
            'stalking_monitoring': [
                # Stalking phrases - highest priority
                'stalking you', 'stalked you', 'following you', 'followed you', 'tracking you',
                'tracked you', 'monitoring you', 'monitored you', 'watching you', 'watched you',
                'spying on you', 'spied on you', 'checking on you', 'checked on you',
                'keep tabs on you', 'keeping tabs on you', 'kept tabs on you',
                'track you down', 'tracking you down', 'tracked you down', 'find you',
                'finding you', 'found you', 'locate you', 'locating you', 'located you',
                'hunt you down', 'hunting you down', 'hunted you down', 'pursue you',
                'pursuing you', 'pursued you', 'chase you', 'chasing you', 'chased you',
                'trail you', 'trailing you', 'trailed you', 'shadow you', 'shadowing you',
                'shadowed you', 'tail you', 'tailing you', 'tailed you',
                # Interrogation phrases
                'where are you', 'where were you', 'where did you go', 'where are you going',
                'where have you been', 'what are you doing', 'what did you do',
                'who were you with', 'who are you with', 'why were you there', 'why did you go',
                'explain yourself', 'account for yourself', 'answer for yourself',
                # Phone monitoring phrases
                'phone tracking', 'gps tracking', 'location tracking', 'phone location',
                'find my phone', 'where is your phone', 'check your phone', 'checking your phone',
                'checked your phone', 'look at your phone', 'looking at your phone',
                'looked at your phone', 'read your messages', 'reading your messages',
                'read your texts', 'reading your texts', 'check your messages',
                'checking your messages', 'checked your messages', 'check your texts',
                'checking your texts', 'checked your texts', 'go through your phone',
                'going through your phone', 'went through your phone', 'search your phone',
                'searching your phone', 'searched your phone',
                # Social media monitoring phrases
                'check social media', 'checking social media', 'checked social media',
                'check your facebook', 'checking your facebook', 'check your instagram',
                'checking your instagram', 'who are you talking to', 'who did you talk to',
                'who are you messaging', 'who did you message', 'who are you texting',
                'who did you text'
            ],
            
            # Pregnancy and child harm - PHRASES ONLY (remove innocent single words)
            'pregnancy_child_harm': [
                # Remove all innocent single words - only keep threatening phrases
                # Threats to pregnancy - phrases only
                'hurt the pregnancy', 'harm the pregnancy', 'damage the pregnancy', 'end the pregnancy',
                'terminate the pregnancy', 'lose the pregnancy', 'kill the pregnancy', 'get rid of the pregnancy',
                'hurt the unborn', 'harm the unborn', 'damage the unborn', 'kill the unborn', 'hurt the fetus',
                'harm the fetus', 'damage the fetus', 'kill the fetus', 'hurt the baby', 'harm the baby',
                'kill the baby', 'lose the baby', 'cause miscarriage', 'make you miscarry', 'force miscarriage',
                'force abortion', 'make you abort', 'force termination',
                'hurt while pregnant', 'harm while pregnant', 'hit while pregnant', 'punch while pregnant',
                'kick while pregnant', 'beat while pregnant', 'hurt when pregnant', 'harm when pregnant',
                'hit when pregnant', 'punch when pregnant', 'kick when pregnant', 'beat when pregnant',
                'you\'ll lose it', 'you\'ll lose the baby', 'you\'ll lose the pregnancy', 'you\'ll miscarry',
                'you\'ll have a miscarriage', 'you won\'t have it', 'you won\'t have the baby',
                'you won\'t have this baby', 'not having it', 'not having the baby', 'not keeping it',
                'not keeping the baby', 'get rid of it', 'get rid of the baby', 'get rid of the pregnancy',
                # Threats to children - phrases only
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
                'no access to kids', 'no access to children',
                # Physical harm to children - phrases only
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
                # Coercive control using children - phrases only
                'using the kids', 'using the children', 'using the child', 'using the baby',
                'through the kids', 'through the children', 'through the child', 'through the baby',
                'using custody', 'threaten custody', 'threatening custody', 'custody battle',
                'custody fight', 'fight for custody', 'take custody', 'get custody', 'win custody',
                'lose custody', 'threaten to take custody', 'threatening to take custody'
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
        Prioritizes phrases over single words to reduce false positives.
        
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
        
        # Track matched positions to avoid double-counting
        matched_positions = set()
        
        for category, word_list in self.word_categories.items():
            category_matches = []
            category_score = 0.0
            
            # Separate phrases (multi-word) from single words
            phrases = [w for w in word_list if ' ' in w or len(w.split()) > 1]
            single_words = [w for w in word_list if w not in phrases]
            
            # Process phrases FIRST (higher priority)
            for phrase in phrases:
                # Use word boundaries for phrase matching
                escaped_phrase = re.escape(phrase.lower())
                pattern = r'\b' + escaped_phrase + r'\b'
                found_matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in found_matches:
                    # Check if this position overlaps with already matched text
                    match_range = range(match.start(), match.end())
                    if any(pos in matched_positions for pos in match_range):
                        continue  # Skip if already matched
                    
                    # Mark positions as matched
                    matched_positions.update(match_range)
                    
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(transcription_text), match.end() + 50)
                    context = transcription_text[start:end].strip()
                    
                    category_matches.append({
                        'word': phrase,
                        'position': match.start(),
                        'context': context,
                        'is_phrase': True
                    })
                    
                    # Phrases get 10x weight multiplier
                    weight = self.category_weights.get(category, 1.0) * 10.0
                    category_score += weight
                    match_count += 1
            
            # Process single words SECOND (lower priority, only if not already matched)
            for word in single_words:
                # Use word boundaries for exact word matching
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                found_matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in found_matches:
                    # Skip if this position is already covered by a phrase match
                    if match.start() in matched_positions:
                        continue
                    
                    # Get context around the match
                    start = max(0, match.start() - 30)
                    end = min(len(transcription_text), match.end() + 30)
                    context = transcription_text[start:end].strip()
                    
                    category_matches.append({
                        'word': word,
                        'position': match.start(),
                        'context': context,
                        'is_phrase': False
                    })
                    
                    # Single words get base weight (no multiplier)
                    weight = self.category_weights.get(category, 1.0)
                    category_score += weight
                    match_count += 1
                    # Mark position (but don't block single word matches if they're different words)
                    matched_positions.add(match.start())
            
            if category_matches:
                category_scores[category] = category_score
                matches[category] = category_matches
                total_score += category_score
        
        # Get top matches (sorted by actual score contribution)
        top_matches = []
        for category, category_matches in matches.items():
            base_weight = self.category_weights.get(category, 1.0)
            for match in category_matches:
                # Calculate actual weight (phrases get 10x multiplier)
                actual_weight = base_weight * (10.0 if match.get('is_phrase', False) else 1.0)
                top_matches.append({
                    'category': category,
                    'word': match['word'],
                    'context': match['context'],
                    'weight': actual_weight,
                    'is_phrase': match.get('is_phrase', False)
                })
        
        # Sort by actual weight (descending) - phrases will rank much higher
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

