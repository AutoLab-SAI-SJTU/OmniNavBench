"""WebRTC handler for receiving video stream.

Handles WebRTC signaling and media track reception.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

logger = logging.getLogger(__name__)


class FrameReceiverTrack(VideoStreamTrack):
    """Video track wrapper that receives frames and calls callback."""
    
    def __init__(self, track: VideoStreamTrack, on_frame: Callable[[np.ndarray], None]):
        """Initialize frame receiver.
        
        Args:
            track: Remote video track to receive from
            on_frame: Callback function(frame: np.ndarray) called for each frame
        """
        super().__init__()
        self.track = track
        self.on_frame = on_frame
    
    async def recv(self):
        """Receive frame from remote track and call callback."""
        frame = await self.track.recv()
        
        # Convert av.VideoFrame to numpy array
        img = frame.to_ndarray(format="rgb24")
        
        # Call callback in background to avoid blocking
        if self.on_frame:
            try:
                self.on_frame(img)
            except Exception as e:
                logger.error(f"Error in frame callback: {e}", exc_info=True)
        
        return frame


class WebRTCHandler:
    """WebRTC connection handler for video streaming."""
    
    def __init__(self, on_frame: Callable[[np.ndarray], None]):
        """Initialize WebRTC handler.
        
        Args:
            on_frame: Callback(frame: np.ndarray) for received frames
        """
        self.on_frame = on_frame
        self.pc: Optional[RTCPeerConnection] = None
        self.receiver_track: Optional[FrameReceiverTrack] = None
    
    async def handle_offer(self, offer_sdp: str) -> str:
        """Handle WebRTC offer and return answer.
        
        Args:
            offer_sdp: SDP offer string
            
        Returns:
            SDP answer string
        """
        # Close existing connection if any
        if self.pc:
            await self.close()
        
        # Create peer connection
        self.pc = RTCPeerConnection()
        
        # Set up track receiver
        @self.pc.on("track")
        def on_track(track):
            """Handle incoming media track."""
            if track.kind == "video":
                self.receiver_track = FrameReceiverTrack(track, self.on_frame)
                self.pc.addTrack(self.receiver_track)
        
        # Handle offer
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await self.pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return self.pc.localDescription.sdp
    
    async def close(self):
        """Close WebRTC connection."""
        if self.pc:
            await self.pc.close()
            self.pc = None
        self.receiver_track = None

