"""
WebSocket Manager for Real-time Updates
Gestionnaire WebSocket pour les mises à jour temps réel
"""

import asyncio
import json
import logging
import websockets
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import socket

from .streaming_engine import (
    get_streaming_engine,
    StreamEvent,
    StreamEventType
)

logger = logging.getLogger(__name__)

class WebSocketMessageType(Enum):
    """Types de messages WebSocket"""
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    EVENT = "event"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

class WebSocketEventChannel(Enum):
    """Canaux d'événements WebSocket"""
    ALL = "all"
    AGENTS = "agents"
    PLANS = "plans"
    PROGRESS = "progress"
    LOGS = "logs"
    METRICS = "metrics"
    NOTIFICATIONS = "notifications"

@dataclass
class WebSocketClient:
    """Client WebSocket connecté"""
    client_id: str
    websocket: Any  # websockets.WebSocketServerProtocol
    subscribed_channels: Set[str]
    connected_at: datetime
    last_ping: datetime
    user_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'client_id': self.client_id,
            'subscribed_channels': list(self.subscribed_channels),
            'connected_at': self.connected_at.isoformat(),
            'last_ping': self.last_ping.isoformat(),
            'user_data': self.user_data
        }

@dataclass
class WebSocketMessage:
    """Message WebSocket structuré"""
    message_id: str
    message_type: WebSocketMessageType
    channel: str
    data: Dict[str, Any]
    timestamp: datetime
    client_id: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps({
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'channel': self.channel,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'client_id': self.client_id
        })
    
    @classmethod
    def from_json(cls, json_str: str, client_id: str = None) -> 'WebSocketMessage':
        data = json.loads(json_str)
        return cls(
            message_id=data.get('message_id', str(uuid.uuid4())),
            message_type=WebSocketMessageType(data['message_type']),
            channel=data['channel'],
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(),
            client_id=client_id
        )

class WebSocketManager:
    """Gestionnaire WebSocket pour les mises à jour temps réel"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketClient] = {}
        self.server = None
        self.server_task = None
        self.streaming_engine = get_streaming_engine()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistiques
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_transferred': 0,
            'start_time': datetime.now()
        }
        
        # Canal de diffusion par défaut
        self.broadcast_channels = {
            WebSocketEventChannel.ALL.value: set(),
            WebSocketEventChannel.AGENTS.value: set(),
            WebSocketEventChannel.PLANS.value: set(),
            WebSocketEventChannel.PROGRESS.value: set(),
            WebSocketEventChannel.LOGS.value: set(),
            WebSocketEventChannel.METRICS.value: set(),
            WebSocketEventChannel.NOTIFICATIONS.value: set()
        }
        
        # S'abonner aux événements du streaming engine
        self._setup_streaming_integration()
        
    def _setup_streaming_integration(self):
        """Configurer l'intégration avec le streaming engine"""
        
        # S'abonner à tous les événements
        self.streaming_engine.subscribe('all', self._handle_streaming_event)
        
    def _handle_streaming_event(self, event: StreamEvent):
        """Gérer un événement du streaming engine"""
        
        # Convertir l'événement en message WebSocket
        ws_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=WebSocketMessageType.EVENT,
            channel=self._map_event_to_channel(event.event_type),
            data=event.to_dict(),
            timestamp=event.timestamp
        )
        
        # Diffuser aux clients abonnés
        asyncio.create_task(self._broadcast_to_channel(ws_message.channel, ws_message))
        
    def _map_event_to_channel(self, event_type: StreamEventType) -> str:
        """Mapper un type d'événement vers un canal WebSocket"""
        
        mapping = {
            StreamEventType.AGENT_START: WebSocketEventChannel.AGENTS.value,
            StreamEventType.AGENT_PROGRESS: WebSocketEventChannel.AGENTS.value,
            StreamEventType.AGENT_COMPLETE: WebSocketEventChannel.AGENTS.value,
            StreamEventType.AGENT_ERROR: WebSocketEventChannel.AGENTS.value,
            StreamEventType.PLAN_CHUNK: WebSocketEventChannel.PLANS.value,
            StreamEventType.PLAN_COMPLETE: WebSocketEventChannel.PLANS.value,
            StreamEventType.STATUS_UPDATE: WebSocketEventChannel.PROGRESS.value,
            StreamEventType.METRICS_UPDATE: WebSocketEventChannel.METRICS.value,
            StreamEventType.LOG_MESSAGE: WebSocketEventChannel.LOGS.value,
            StreamEventType.SYSTEM_MESSAGE: WebSocketEventChannel.NOTIFICATIONS.value
        }
        
        return mapping.get(event_type, WebSocketEventChannel.ALL.value)
        
    async def start_server(self):
        """Démarrer le serveur WebSocket"""
        
        try:
            # Vérifier si le port est disponible
            if self._is_port_in_use(self.port):
                logger.warning(f"Port {self.port} déjà utilisé, tentative avec port alternatif")
                self.port = self._find_available_port()
                
            self.server = await websockets.serve(
                self._handle_client_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            logger.info(f"🌐 Serveur WebSocket démarré sur {self.host}:{self.port}")
            
            # Démarrer les tâches de maintenance
            asyncio.create_task(self._heartbeat_task())
            asyncio.create_task(self._cleanup_task())
            
        except Exception as e:
            logger.error(f"Erreur démarrage serveur WebSocket: {e}")
            raise
            
    def _is_port_in_use(self, port: int) -> bool:
        """Vérifier si un port est déjà utilisé"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, port)) == 0
            
    def _find_available_port(self, start_port: int = 8765, max_attempts: int = 100) -> int:
        """Trouver un port disponible"""
        for port in range(start_port, start_port + max_attempts):
            if not self._is_port_in_use(port):
                return port
        raise RuntimeError("Aucun port disponible trouvé")
        
    async def _handle_client_connection(self, websocket, path):
        """Gérer une nouvelle connexion client"""
        
        client_id = str(uuid.uuid4())
        
        try:
            # Créer le client
            client = WebSocketClient(
                client_id=client_id,
                websocket=websocket,
                subscribed_channels=set(),
                connected_at=datetime.now(),
                last_ping=datetime.now(),
                user_data={}
            )
            
            self.clients[client_id] = client
            self.stats['total_connections'] += 1
            self.stats['active_connections'] += 1
            
            logger.info(f"💫 Nouveau client WebSocket connecté: {client_id}")
            
            # Envoyer message de bienvenue
            welcome_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=WebSocketMessageType.STATUS,
                channel="system",
                data={
                    'status': 'connected',
                    'client_id': client_id,
                    'server_info': {
                        'host': self.host,
                        'port': self.port,
                        'available_channels': list(self.broadcast_channels.keys())
                    }
                },
                timestamp=datetime.now()
            )
            
            await self._send_message(client_id, welcome_message)
            
            # Boucle de traitement des messages
            async for message in websocket:
                await self._handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client WebSocket déconnecté: {client_id}")
        except Exception as e:
            logger.error(f"Erreur client WebSocket {client_id}: {e}")
        finally:
            # Nettoyage
            await self._cleanup_client(client_id)
            
    async def _handle_client_message(self, client_id: str, raw_message: str):
        """Gérer un message reçu d'un client"""
        
        try:
            message = WebSocketMessage.from_json(raw_message, client_id)
            self.stats['messages_received'] += 1
            self.stats['bytes_transferred'] += len(raw_message)
            
            # Traiter selon le type de message
            if message.message_type == WebSocketMessageType.SUBSCRIBE:
                await self._handle_subscribe(client_id, message)
            elif message.message_type == WebSocketMessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(client_id, message)
            elif message.message_type == WebSocketMessageType.PING:
                await self._handle_ping(client_id, message)
            elif message.message_type == WebSocketMessageType.STATUS:
                await self._handle_status_request(client_id, message)
            else:
                logger.warning(f"Type de message non géré: {message.message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Message JSON invalide reçu de {client_id}: {raw_message}")
            await self._send_error(client_id, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Erreur traitement message de {client_id}: {e}")
            await self._send_error(client_id, str(e))
            
    async def _handle_subscribe(self, client_id: str, message: WebSocketMessage):
        """Gérer un abonnement à un canal"""
        
        channel = message.data.get('channel', 'all')
        
        if client_id in self.clients:
            client = self.clients[client_id]
            client.subscribed_channels.add(channel)
            
            # Ajouter aux canaux de diffusion
            if channel in self.broadcast_channels:
                self.broadcast_channels[channel].add(client_id)
            else:
                # Créer nouveau canal si n'existe pas
                self.broadcast_channels[channel] = {client_id}
                
            # Confirmation d'abonnement
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=WebSocketMessageType.STATUS,
                channel=channel,
                data={
                    'status': 'subscribed',
                    'channel': channel,
                    'total_subscribers': len(self.broadcast_channels[channel])
                },
                timestamp=datetime.now()
            )
            
            await self._send_message(client_id, response)
            logger.debug(f"Client {client_id} abonné au canal {channel}")
            
    async def _handle_unsubscribe(self, client_id: str, message: WebSocketMessage):
        """Gérer un désabonnement d'un canal"""
        
        channel = message.data.get('channel', 'all')
        
        if client_id in self.clients:
            client = self.clients[client_id]
            client.subscribed_channels.discard(channel)
            
            # Retirer des canaux de diffusion
            if channel in self.broadcast_channels:
                self.broadcast_channels[channel].discard(client_id)
                
            # Confirmation de désabonnement
            response = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=WebSocketMessageType.STATUS,
                channel=channel,
                data={
                    'status': 'unsubscribed',
                    'channel': channel
                },
                timestamp=datetime.now()
            )
            
            await self._send_message(client_id, response)
            logger.debug(f"Client {client_id} désabonné du canal {channel}")
            
    async def _handle_ping(self, client_id: str, message: WebSocketMessage):
        """Gérer un ping client"""
        
        if client_id in self.clients:
            self.clients[client_id].last_ping = datetime.now()
            
        # Répondre avec pong
        pong_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=WebSocketMessageType.PONG,
            channel="system",
            data={'timestamp': datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        await self._send_message(client_id, pong_message)
        
    async def _handle_status_request(self, client_id: str, message: WebSocketMessage):
        """Gérer une demande de status"""
        
        status_data = {
            'server_stats': self.stats.copy(),
            'connected_clients': len(self.clients),
            'active_channels': {k: len(v) for k, v in self.broadcast_channels.items()},
            'streaming_status': self.streaming_engine.get_streaming_status()
        }
        
        response = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=WebSocketMessageType.STATUS,
            channel="system",
            data=status_data,
            timestamp=datetime.now()
        )
        
        await self._send_message(client_id, response)
        
    async def _send_message(self, client_id: str, message: WebSocketMessage):
        """Envoyer un message à un client spécifique"""
        
        if client_id not in self.clients:
            return
            
        try:
            client = self.clients[client_id]
            message_json = message.to_json()
            
            await client.websocket.send(message_json)
            
            self.stats['messages_sent'] += 1
            self.stats['bytes_transferred'] += len(message_json)
            
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client {client_id} déconnecté lors de l'envoi")
            await self._cleanup_client(client_id)
        except Exception as e:
            logger.error(f"Erreur envoi message à {client_id}: {e}")
            
    async def _send_error(self, client_id: str, error_message: str):
        """Envoyer un message d'erreur à un client"""
        
        error_msg = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=WebSocketMessageType.ERROR,
            channel="system",
            data={'error': error_message},
            timestamp=datetime.now()
        )
        
        await self._send_message(client_id, error_msg)
        
    async def _broadcast_to_channel(self, channel: str, message: WebSocketMessage):
        """Diffuser un message à tous les clients d'un canal"""
        
        if channel not in self.broadcast_channels:
            return
            
        clients_to_send = self.broadcast_channels[channel].copy()
        
        # Envoyer à tous les clients du canal
        for client_id in clients_to_send:
            await self._send_message(client_id, message)
            
        # Diffuser aussi au canal 'all' si ce n'est pas déjà fait
        if channel != WebSocketEventChannel.ALL.value:
            all_clients = self.broadcast_channels.get(WebSocketEventChannel.ALL.value, set())
            for client_id in all_clients:
                if client_id not in clients_to_send:  # Éviter les doublons
                    await self._send_message(client_id, message)
                    
    async def _broadcast_to_all(self, message: WebSocketMessage):
        """Diffuser un message à tous les clients connectés"""
        
        for client_id in list(self.clients.keys()):
            await self._send_message(client_id, message)
            
    async def _heartbeat_task(self):
        """Tâche de heartbeat pour maintenir les connexions"""
        
        while True:
            try:
                heartbeat_message = WebSocketMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=WebSocketMessageType.HEARTBEAT,
                    channel="system",
                    data={
                        'timestamp': datetime.now().isoformat(),
                        'active_clients': len(self.clients),
                        'uptime': (datetime.now() - self.stats['start_time']).total_seconds()
                    },
                    timestamp=datetime.now()
                )
                
                await self._broadcast_to_all(heartbeat_message)
                
                # Attendre 30 secondes
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Erreur heartbeat: {e}")
                await asyncio.sleep(30)
                
    async def _cleanup_task(self):
        """Tâche de nettoyage des connexions obsolètes"""
        
        while True:
            try:
                current_time = datetime.now()
                clients_to_remove = []
                
                for client_id, client in self.clients.items():
                    # Supprimer clients inactifs depuis plus de 5 minutes
                    if (current_time - client.last_ping).total_seconds() > 300:
                        clients_to_remove.append(client_id)
                        
                for client_id in clients_to_remove:
                    logger.info(f"Suppression client inactif: {client_id}")
                    await self._cleanup_client(client_id)
                    
                # Attendre 60 secondes
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Erreur tâche de nettoyage: {e}")
                await asyncio.sleep(60)
                
    async def _cleanup_client(self, client_id: str):
        """Nettoyer un client déconnecté"""
        
        if client_id not in self.clients:
            return
            
        try:
            client = self.clients[client_id]
            
            # Retirer de tous les canaux
            for channel in list(client.subscribed_channels):
                if channel in self.broadcast_channels:
                    self.broadcast_channels[channel].discard(client_id)
                    
            # Supprimer le client
            del self.clients[client_id]
            self.stats['active_connections'] -= 1
            
            logger.debug(f"Client {client_id} nettoyé")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage client {client_id}: {e}")
            
    async def stop_server(self):
        """Arrêter le serveur WebSocket"""
        
        if self.server:
            logger.info("🛑 Arrêt du serveur WebSocket...")
            
            # Notifier tous les clients
            shutdown_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=WebSocketMessageType.STATUS,
                channel="system",
                data={'status': 'server_shutdown'},
                timestamp=datetime.now()
            )
            
            await self._broadcast_to_all(shutdown_message)
            
            # Fermer le serveur
            self.server.close()
            await self.server.wait_closed()
            
            logger.info("✅ Serveur WebSocket arrêté")
            
    def get_connection_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de connexion"""
        
        stats = self.stats.copy()
        stats['clients'] = {
            client_id: client.to_dict() 
            for client_id, client in self.clients.items()
        }
        stats['channels'] = {
            channel: len(clients) 
            for channel, clients in self.broadcast_channels.items()
        }
        
        return stats

# Instance globale
global_websocket_manager: Optional[WebSocketManager] = None

def get_websocket_manager(host: str = "localhost", port: int = 8765) -> WebSocketManager:
    """Obtenir l'instance globale du gestionnaire WebSocket"""
    global global_websocket_manager
    
    if global_websocket_manager is None:
        global_websocket_manager = WebSocketManager(host, port)
        
    return global_websocket_manager

async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """Fonction utilitaire pour démarrer le serveur WebSocket"""
    manager = get_websocket_manager(host, port)
    await manager.start_server()
    return manager

def run_websocket_server_in_thread(host: str = "localhost", port: int = 8765):
    """Exécuter le serveur WebSocket dans un thread séparé"""
    
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        manager = get_websocket_manager(host, port)
        
        try:
            loop.run_until_complete(manager.start_server())
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Arrêt du serveur WebSocket...")
        finally:
            loop.run_until_complete(manager.stop_server())
            loop.close()
            
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    return thread

# Client WebSocket simple pour tests
class SimpleWebSocketClient:
    """Client WebSocket simple pour tests"""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Se connecter au serveur"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            logger.info(f"Connecté au serveur WebSocket: {self.uri}")
        except Exception as e:
            logger.error(f"Erreur connexion WebSocket: {e}")
            
    async def subscribe(self, channel: str):
        """S'abonner à un canal"""
        if not self.is_connected:
            return
            
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=WebSocketMessageType.SUBSCRIBE,
            channel="system",
            data={'channel': channel},
            timestamp=datetime.now()
        )
        
        await self.websocket.send(message.to_json())
        
    async def listen(self):
        """Écouter les messages"""
        if not self.is_connected:
            return
            
        try:
            async for message in self.websocket:
                data = json.loads(message)
                logger.info(f"Message reçu: {data}")
                yield data
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connexion WebSocket fermée")
            self.is_connected = False
            
    async def disconnect(self):
        """Se déconnecter"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False