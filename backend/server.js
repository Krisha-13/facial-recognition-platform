const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const sqlite3 = require('sqlite3').verbose();
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));

// Create uploads directory
if (!fs.existsSync('uploads')) {
  fs.mkdirSync('uploads');
}

// Database setup
const db = new sqlite3.Database('../database/faces.db');

// Create faces table
db.run(`
  CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding TEXT NOT NULL,
    image_path TEXT,
    registration_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
  )
`);

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

// Routes

// Get all registered faces
app.get('/api/faces', (req, res) => {
  db.all('SELECT * FROM faces ORDER BY registration_timestamp DESC', (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
      return;
    }
    res.json(rows);
  });
});

// Register a new face
app.post('/api/register-face', upload.single('image'), async (req, res) => {
  try {
    const { name } = req.body;
    const imagePath = req.file.path;

    // Call Python face recognition service
    const response = await axios.post('http://localhost:5001/register', {
      name: name,
      image_path: imagePath
    });

    if (response.data.success) {
      // Store in database
      db.run(
        'INSERT INTO faces (name, encoding, image_path) VALUES (?, ?, ?)',
        [name, JSON.stringify(response.data.encoding), imagePath],
        function(err) {
          if (err) {
            res.status(500).json({ error: err.message });
            return;
          }
          res.json({ 
            success: true, 
            id: this.lastID,
            message: 'Face registered successfully' 
          });
        }
      );
    } else {
      res.status(400).json({ error: response.data.error });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get face encodings for recognition
app.get('/api/encodings', (req, res) => {
  db.all('SELECT id, name, encoding FROM faces', (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
      return;
    }
    
    const encodings = rows.map(row => ({
      id: row.id,
      name: row.name,
      encoding: JSON.parse(row.encoding)
    }));
    
    res.json(encodings);
  });
});

// Socket.io for chat
io.on('connection', (socket) => {
  console.log('User connected');

  socket.on('chat_message', async (data) => {
    try {
      // Forward message to RAG engine
      const response = await axios.post('http://localhost:5002/query', {
        message: data.message
      });

      socket.emit('chat_response', {
        message: response.data.response
      });
    } catch (error) {
      socket.emit('chat_response', {
        message: 'Sorry, I encountered an error processing your request.'
      });
    }
  });

  socket.on('disconnect', () => {
    console.log('User disconnected');
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});