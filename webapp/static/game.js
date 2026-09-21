// webapp/static/game.js

const PIECES = {
    wK:'♔', wQ:'♕', wR:'♖', wB:'♗', wN:'♘', wP:'♙',
    bK:'♚', bQ:'♛', bR:'♜', bB:'♝', bN:'♞', bP:'♟'
  };
  
  let fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  let selectedSquare = null;
  let legalMoves = [];
  let flipped = false;
  let lastMove = null;
  let moveHistory = [];
  
  function parseFen(fen) {
    const parts = fen.split(' ');
    const rows = parts[0].split('/');
    const board = [];
    for (const row of rows) {
      for (const ch of row) {
        if ('12345678'.includes(ch)) board.push(...Array(+ch).fill(null));
        else board.push(ch);
      }
    }
    return { squares: board, turn: parts[1] };
  }
  
  function pieceKey(ch) {
    if (!ch) return null;
    const color = ch === ch.toUpperCase() ? 'w' : 'b';
    return color + ch.toUpperCase();
  }
  
  function renderBoard() {
    const el = document.getElementById('board');
    const { squares, turn } = parseFen(fen);
    el.innerHTML = '';
  
    for (let i = 0; i < 64; i++) {
      const idx = flipped ? (63 - i) : i;
      const row = Math.floor(idx / 8);
      const col = idx % 8;
      const file = 'abcdefgh'[col];
      const rank = 8 - row;
      const sqName = file + rank;
      const piece = squares[idx];
      const key = pieceKey(piece);
  
      const div = document.createElement('div');
      div.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
      div.dataset.square = sqName;
  
      if (lastMove?.includes(sqName)) div.classList.add('last-move');
      if (selectedSquare === sqName) div.classList.add('selected');
  
      const legal = legalMoves.find(m => m.slice(2) === sqName);
      if (legal) div.classList.add(piece ? 'legal-capture' : 'legal-dot');
  
      if (key) {
        const span = document.createElement('span');
        span.textContent = PIECES[key];
        span.className = key.startsWith('w') ? 'piece-white' : 'piece-black';
        div.appendChild(span);
      }
  
      div.addEventListener('click', () => handleSquareClick(sqName, piece, turn));
      el.appendChild(div);
    }
  }
  
  async function handleSquareClick(sq, piece, turn) {
    const isMyTurn = turn === 'w';   // player always plays White
    if (!isMyTurn) return;
  
    if (selectedSquare && legalMoves.find(m => m.slice(2) === sq)) {
      await makeMove(selectedSquare + sq);
      return;
    }
  
    if (piece && piece === piece.toUpperCase()) {
      selectedSquare = sq;
      const res = await fetch('/legal_moves', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ fen, square: sq })
      });
      const data = await res.json();
      legalMoves = data.legal_moves;
    } else {
      selectedSquare = null;
      legalMoves = [];
    }
    renderBoard();
  }
  
  async function makeMove(uci) {
    // Apply player's move via FEN update
    const { squares } = parseFen(fen);
    lastMove = [uci.slice(0, 2), uci.slice(2, 4)];
    selectedSquare = null;
    legalMoves = [];
    moveHistory.push(uci);
    updateMoveList();
  
    // Ask the server to apply move and get the new FEN + bot reply
    document.getElementById('status').textContent = 'Bot is thinking...';
  
    const res = await fetch('/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ fen, player_move: uci })
    });
    const data = await res.json();
  
    if (data.move) {
      lastMove = [data.move.slice(0,2), data.move.slice(2,4)];
      moveHistory.push(data.move);
      updateMoveList();
    }
  
    fen = data.fen || fen;
    document.getElementById('status').textContent =
      data.is_game_over ? `Game over: ${data.result}` : 'Your turn (White)';
  
    renderBoard();
  }
  
  async function resetGame() {
    const res = await fetch('/reset', { method: 'POST' });
    const data = await res.json();
    fen = data.fen;
    selectedSquare = null; legalMoves = []; lastMove = null; moveHistory = [];
    document.getElementById('status').textContent = 'Your turn (White)';
    renderBoard();
  }
  
  function flipBoard() { flipped = !flipped; renderBoard(); }
  
  function updateMoveList() {
    document.getElementById('move-list').textContent = moveHistory.join('  ');
  }
  
  renderBoard();