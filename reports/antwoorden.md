# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?
- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?

# ML: Antwoord 1a
Het lijkt een regressie probleem doordat er onder andere cijfers voorspeld dienen te worden, maar het betreft een classificatie probleem, het voorspellen van 20 classes: cijfers 0 t/m 9 van mannen en vrouwen.

Een neuraal netwerk met meerdere lineaire lagen is een goede architectuur om mee te starten (baseline). Dit type model is geschikt voor zowel regressie als classificatie problemen en wordt veelal gebruikt voor beeld- en taalclassificatie.

Sterke punten:
  * Ideaal om te gebruiken als baselinemodel.
  * Relatief simpel model dat uitlegbaar is.
  * Weinig lagen, dus minder complexiteit.
  * Eenvoudigheid van het model helpt bij het voorkomen van overfitten.
  * Goed toepasbaar op minder complexe problemen.

Zwakke punten:
  * Door eenvoudigheid gaat deze underfitten op non-lineaire datasets. 
  * Accuracy zal met dit model niet gemakkelijk te verhogen zijn doordat er relatief weinig features te trainen zijn.
  * Werkt niet voor grote hoeveelheden data die veel features hebben.

Gezien de gesproken cijfers met tijd te maken hebben gaat het om timeseries waarbij de volgordelijkheid erg van belang is. Een model met 'geheugen' zou beter zijn gezien de volgordelijkheid van timeseries: denk hierbij aan Simple RNN, LSTM en GRU. Een Simple RNN model heeft hoogstwaarschijnlijk moeite met de volgordelijkheid, gezien het beperkte 'geheugen'. Hierdoor zou een LSTM of GRU model betere resultaten kunnen behalen.

Keuze aantal units ten opzichte van de data:
- input = 13
- hidden1 = 100
- hidden2 = 10
- output = 20
- dropout = 0,5
- Dataset = 8800 (10 digits x 10 repetitions x 88 speakers) van 44 mannen en 44 vrouwen. Batchsize = 128

Input komt overeen met het aantal attributen: 13. Deze attributen bestaan uit regels met timeseries van 13 Frequency Cepstral Coefficients (MFCCs).

Door het aantal units als eerste te vergroten, vervolgens te verkleinen en dan weer te vergroten om terug te gaan naar 20 classes is niet logisch. Het is logischer om van groot naar klein te werken, hierdoor dient de eerst laag het grootst te zijn. In deze situatie is de input 13 units, deze te vergroten naar een veelvoud van twee. Denk hierbij aan de volgende reeks die we ook hebben toegepast tijdens het handmatig hypertunen. Hierdoor is gemakkelijker om een passende range te vinden waarbij de units optimaal zijn: 8, 16, 32, 64, 128, 256 en 512. Veel meer units zal niet nodig zijn gezien de beperkte grote van de dataset. Als baseline zou ik in dit geval de eerste hidden layer de op 128 units hebben gezet, de tweede layer dus kleiner maken en op 64 units hebben gezet aangezien de output layer terug gaat naar 20 classes.

Daarnaast wordt er gebruik gemaakt van een dropout van 0,5 wat voor een relatief kleine dataset aan de hoge kant is. Door de dropout wordt er random in iedere batch de helft van de gegevens tussen de hiddenlayers weggegooid. Hierdoor kunnen units die aan het overfitten zijn verminderd worden, wat het risico op overfitten verkleind. Ik zou op basis van onderbuik gevoel de dropout verlagen naar 0,3. Echter is het erg afhankelijk van de kenmerken van de dataset welke dropout het beste werkt, daarom is het goed om te gaan experimenteren met de waarde van de dropout.

Mijn initiele model zou de volgende settings krijgen:
- input = 13
- hidden1 = 128
- hidden2 = 64
- output = 20
- dropout = 0,3

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
- Hoe had hij dit ook kunnen oplossen?
- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

# ML: Antwoord 1b
Het effect van 'x.mean(dim=1)' is dat het aantal dimensies wordt gereduceerd. De tijdsdimensie kan verschillende lengtes hebben, het ene cijfer duurt langer om uit gesproken te worden dan de ander. Door alleen het gemiddelde te nemen wordt de tijdsdimensie gereduceerd, hadden we dit niet gedaan dan had het neurale netwerk de gehele tijdserie van elke sample moeten verwerken wat er voorzorgt dat het trainingsproces veel langzamer wordt. Het netwerk krijgt te veel features om te verwerken waardoor het moeilijker wordt om van de data te leren. Tevens wordt de het toepassen van x.mean(dim=1) ook de ruis verminderd, die ontstaat door de verschillende lengtes van de tijdsdimensie.

Een alternatief voor het reduceren van de dimensies is het toepassen van een flatten. Door flatten wordt alle informatie uit de dimensies platgeslagen tot een reeks. Echter door het toepassen van flatten gaan er veel kenmerken van tijdserie verloren. Daarom is het toepassen van flatten niet optimaal voor het verwerken van tijdseries.

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.
- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.
- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

# ML: Antwoord 1c
De verschillende architecturen die dit problemen zijn zoals eerder benoemd de volgende:
- Simple RNN: Basis model, echter moeite met geheugen.
- LSTM: Heeft meer parameters met 3 gates en 2 hidden states, dus een meer complex model.
- GRU: Versimpelde versie van LSTM met 2 gates en 1 hidden states.

<figure>
  <p align = "center">
    <img src="img/Architecturen.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 1c. Overzicht Architecturen</b>
    </figcaption>
  </p>
</figure>

Deze architecturen zijn het best toepasbaar voor timeseries data aangezien ze beschikken over geheugen.

In deze casus is mijn verwachting dat het GRU model het beste is, gezien dat een Simple RNN beperking in het gebruik van geheugen heeft. Een LSTM model is te complex voor deze relatief simpele dataset, echter is het geheugen wel veel beter ten opzichte van een Simple RNN Model. Gezien GRU de versimpelde versie van LSTM, zie ik GRU als het meest veelbelovende model.

Educated guess: op basis van oefening les, onderbuikt.. NOG VERDER TOELICHTEN!
- Hidden size: 64
  - Gezien er meerdere lagen in het model zitten had ik een onderbuik gevoel dat de hidden size relatief klein kon zijn.
- Aantal lagen: 4
  - Op basis van de oefeningen uit de les in combinatie met compacte dataset zou dit voldoende moeten zijn met 4 lagen die verder getraind kunnen worden.
- Dropout: 0.3
  - Geen dropout van 0.5 doordat er dan mogelijk te veel data van de kleine dataset wordt weggegooid.
- Batchsize: 128 
  - (default van eerste model, zal in hypertunen gaan varieren)
- Learning rate: 1e-3
  - Gedurende de module de best werkende learningrate.
- Optimizer: Adam
  - Vaak de beste optimzer in combinatie met learning rate van 1e -3.
- Loss functie: CrossEntropyLoss
  - Gezien het om classificatie is deze loss functie het beste.

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

# ML: Antwoord 1d
In model.py file heb ik nieuw model toegevoegd: class 'GRUmodel', hierin is de gebruikte architectuur te vinden. Tevens heb ik in de settings.py file de config voor het model toegevoegd: class 'GRUmodelConfig', hierin worden het type van de verwachtte input geconfigureerd. Om het GRUmodel te runnen en de architectuur van het initiele model(01_model_design.py) te behouden heb ik een nieuw script aangemaakt (01_model_GRU_design.py). Om dit model losstaand van het eerste model te laten runnen heb ik de Makefile moeten aanpassen zodat ik een 'make' commando kon geven voor alle dit model: 'make runGRU'. Voor het handmatig hypertunen ben ik gaan experimenteren met de volgende parameters: hidden_size, aantal lagen en dropout. De overige parameters hebben de volgende default settings van mij meegekregen:
- Batchsize = 128
- Learningrate = 1e -3
- Optimizer = Adam
- Loss = CrossEntropyLoss

### Test1d.1 Logs/20230121-2220
Zoals in in vraag 1c benoemd heb ik in eerste instantie het model gerund met de educated guess: 
- Hidden size = 64
- Aantal lagen = 4
- Dropout = 0.3

Na ongeveer 25/30 epochs is er al een accuracy van 95% behaald.
Learningrate valt na 40 epochts helemaal terug, wat betekent dat het model op die learningrate niets meer aan het leren was.
Model is niet aan het overfitten: Zowel de Loss/test als de Loss/train curve zitten op hetzelfde niveau.

### Test1d.2
- Hidden size = 128
- Aantal lagen = 6
- Dropout = 0.3



## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je je model aanpast, maak dan even een tweede model aan. Pas je een dataloader aan, maak dan een nieuwe dataloader > Benoemd in de les toevoeging Raoul
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash course.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
