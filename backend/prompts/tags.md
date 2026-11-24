# Tags

Controlled vocabulary for tagging the sections of the commentary. A tagger may only use the
tags listed here. Each `##` heading is one facet:

- `type: choice` — exactly one tag per section (the dominant one).
- `type: noul` — every tag is a separate yes/no question; a section can get several or none.

The text after the colon tells the model what the tag means; keep it short and concrete.
Tag names are the keys used in the data and the reader, so rename them only deliberately.

## genre
type: choice
instructions: What does the author mainly do in this passage?
- exegese: explains the meaning of the biblical text verse by verse, or retells and clarifies the story
- filologie: discusses Hebrew, Greek or Latin words, etymology, translations and readings of the text
- quaestio: poses a theological or moral question and argues for an answer
- polemiek: refutes the views of named opponents, such as papists, Jesuits or Jews
- toepassing: draws lessons, doctrines or exhortations for faith and life, for preachers and readers
- realia: discusses history, geography, natural history, medicine or customs behind the text

## loci
type: noul
instructions: Does this passage substantially discuss {tag}?
- god: God, his names (Jehovah, El Shaddai) and his attributes
- voorzienigheid: God's providence and his governing of events and nations
- verharding: the hardening of Pharaoh's heart, God's role in sin and evil
- wonderen: miracles, signs and plagues, and what they prove
- roeping: calling, mission and office of prophets, ministers or leaders
- kerk: the true church, its marks and its reformation
- verbond: God's covenant and promises to the patriarchs and to Israel
- wet: the law of Moses, commandments and ceremonial laws
- sabbat: the sabbath and the keeping of holy days
- pascha: the Passover, the unleavened bread and the feasts of Israel
- sacramenten: sacraments such as circumcision, baptism and the Lord's Supper
- christus: Christ, and types or figures that point to him
- eredienst: worship, prayer, sacrifice and idolatry
- tovenarij: magicians, sorcery, demons and angels
- overheid: rulers, obedience, rebellion, lying and deception toward authorities
- eeuwig-leven: heaven, eternal life and the last things

## tegenstanders
type: noul
instructions: Does the author argue against {tag}?
- rooms: Roman Catholics, papists, Jesuits such as Bellarmine or Pererius
- joods: Jewish interpreters, rabbis or Jewish customs
- heidens: pagan authors who slander the Jews or the Bible, such as Tacitus
- ketters: Socinians, Anabaptists, Arminians or other heretics

## bronnen
type: noul
instructions: Does the passage quote or discuss {tag}?
- kerkvaders: the Church Fathers, such as Augustine, Jerome or Chrysostom
- rabbijnen: rabbinic writers such as Aben Ezra, Rashi, Kimchi, or the Talmud
- vertalingen: ancient translations: the Septuagint, the Vulgate, the Chaldee paraphrase
- klassiek: classical Greek and Latin authors, such as Pliny, Tacitus or Herodotus
- josephus: Josephus or Philo
- reformatoren: Protestant authors, such as Calvin, Junius, Tremellius or Beza
