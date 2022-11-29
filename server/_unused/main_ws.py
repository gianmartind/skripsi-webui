import asyncio
import websockets
from client_ws import ClientSocket

id = 0
clients = dict()

async def handler(websockets):
    while True:
        message = await websockets.recv()
        print(message)
        await websockets.send('Hello!')

async def main():
    async with websockets.serve(handler, 'localhost', 1113):
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())