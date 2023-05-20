# cupidshuffle
Down, down, do your dance, do your dance

![](memes/cupid-shuffle.gif)

```python
from model import CupidShuffle

net = CupidShuffle(start_channels=28, token_dim=28, repeats=[1, 4, 1])

output = net(image)
```