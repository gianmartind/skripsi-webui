import asyncio

class ClientSocket():
    def __init__(self, websocket):
        self.websocket = websocket
        self.loop = asyncio.get_event_loop()

    def run(self):
        return self.loop.run_until_complete(self.event_loop())
    
    async def event_loop(self):
        await self.receive()
        await self.send()

    async def receive(self):
        message = await self.websocket.recv()
        print(message)
    
    async def send(self):
        await self.websocket.send('Hello!')
