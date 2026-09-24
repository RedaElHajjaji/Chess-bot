// webapp/static/game.js

// ── Piece unicode map ──
const PIECES = {
  wK:'♔', wQ:'♕', wR:'♖', wB:'♗', wN:'♘', wP:'♙',
  bK:'♚', bQ:'♛', bR:'♜', bB:'♝', bN:'♞', bP:'♟'
};

// ── State ──
let fen            = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
let selectedSquare = null;
let legalMoves     = [];
let lastMove       = [];
let flipped        = false;
let moveHistory    = [];
let personality    = 'balanced';
let gameOver       = false;
let inCheckSquare  = null;
let moveCount      = 0;

// ── Parse FEN into 64-square array ──
function parseFen(fen) {
  const parts  = fen.split(' ');
  const rows   = parts[0].split('/');
  const board  = [];
  const turn   = parts[1];

  for (const row of rows) {
    for (const ch of row) {
      if ('12345678'.includes(ch)) {
        board.push(...Array(+ch).fill(null));
      } else {
        board.push(ch);
      }
    }
  }
  return { squares: board, turn };
}

// ── Get piece key (e.g. 'wK', 'bP') ──
function pieceKey(ch) {
  if (!ch) return null;
  return (ch === ch.toUpperCase() ? 'w' : 'b') + ch.toUpperCase();
}

// ── Render the board ──
function renderBoard() {
  const el = document.getElementById('board');
  const { squares, turn } = parseFen(fen);
  el.innerHTML = '';

  for (let i = 0; i < 64; i++) {
    const idx  = flipped ? (63 - i) : i;
    const row  = Math.floor(idx / 8);
    const col  = idx % 8;
    const file = 'abcdefgh'[col];
    const rank = 8 - row;
    const sqName = file + rank;
    const piece  = squares[idx];
    const key    = pieceKey(piece);

    const div = document.createElement('div');
    div.className = `square ${(row + col) % 2 === 0 ? 'light' : 'dark'}`;
    div.dataset.square = sqName;

    // Highlights
    if (lastMove.includes(sqName))    div.classList.add('last-move');
    if (selectedSquare === sqName)    div.classList.add('selected');
    if (inCheckSquare === sqName)     div.classList.add('in-check');

    // Legal move indicators
    const legalMove = legalMoves.find(m => m.slice(2,4) === sqName);
    if (legalMove) {
      div.classList.add(piece ? 'legal-capture' : 'legal-dot');
    }

    // Piece
    if (key) {
      const span = document.createElement('span');
      span.textContent = PIECES[key];
      span.className   = key.startsWith('w') ? 'piece-white' : 'piece-black';
      div.appendChild(span);
    }

    div.addEventListener('click', () =>
      handleSquareClick(sqName, piece, turn));
    el.appendChild(div);
  }

  // Update coordinates if flipped
  updateCoords();
}

function updateCoords() {
  const ranks = flipped
    ? ['1','2','3','4','5','6','7','8']
    : ['8','7','6','5','4','3','2','1'];
  const files = flipped
    ? ['h','g','f','e','d','c','b','a']
    : ['a','b','c','d','e','f','g','h'];

  const leftEl   = document.getElementById('coords-left');
  const bottomEl = document.getElementById('coords-bottom');

  leftEl.innerHTML   = ranks.map(r => `<span>${r}</span>`).join('');
  bottomEl.innerHTML = files.map(f => `<span>${f}</span>`).join('');
}

// ── Handle square click ──
async function handleSquareClick(sq, piece, turn) {
  if (gameOver) return;
  if (turn !== 'w') return;   // player always plays White

  // If a square is selected and this is a legal destination — make the move
  if (selectedSquare && legalMoves.find(m => m.slice(2,4) === sq)) {
    const uci = selectedSquare + sq;

    // Handle pawn promotion — always promote to queen
    const { squares } = parseFen(fen);
    const fromIdx = ('abcdefgh'.indexOf(selectedSquare[0])) +
                    (8 - parseInt(selectedSquare[1])) * 8;
    const fromPiece = squares[fromIdx];

    let fullUci = uci;
    if (fromPiece === 'P' && sq[1] === '8') fullUci = uci + 'q';
    if (fromPiece === 'p' && sq[1] === '1') fullUci = uci + 'q';

    selectedSquare = null;
    legalMoves     = [];
    renderBoard();
    await makeMove(fullUci);
    return;
  }

  // Select a white piece
  if (piece && piece === piece.toUpperCase()) {
    selectedSquare = sq;
    const res = await fetch('/legal_moves', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ fen, square: sq })
    });
    const data = await res.json();
    legalMoves = data.legal_moves || [];
  } else {
    selectedSquare = null;
    legalMoves     = [];
  }

  renderBoard();
}

// ── Make a move ──
async function makeMove(uci) {
  if (gameOver) return;

  // Update last move highlight
  lastMove       = [uci.slice(0,2), uci.slice(2,4)];
  inCheckSquare  = null;

  // Add to move history
  moveCount++;
  const isWhite = moveCount % 2 === 1;
  const moveNum = Math.ceil(moveCount / 2);
  moveHistory.push({ uci, isWhite, moveNum });
  updateMoveList();

  // Show thinking indicator
  setStatus('thinking');

  try {
    const res = await fetch('/move', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ fen, player_move: uci, personality })
    });

    const data = await res.json();

    if (data.error) {
      console.error('Move error:', data.error);
      setStatus('your-turn');
      return;
    }

    // Record bot move
    if (data.move) {
      lastMove = [data.move.slice(0,2), data.move.slice(2,4)];
      moveCount++;
      moveHistory.push({
        uci:     data.move,
        isWhite: false,
        moveNum: Math.ceil(moveCount / 2)
      });
      updateMoveList();
    }

    fen = data.fen || fen;

    // Handle check
    if (data.in_check && data.king_square) {
      inCheckSquare = data.king_square;
    }

    // Handle game over
    if (data.is_game_over) {
      gameOver = true;
      renderBoard();
      showResult(data.result, data.reason);
      setStatus('gameover');
      return;
    }

    setStatus('your-turn');
    renderBoard();

  } catch (err) {
    console.error('Fetch error:', err);
    setStatus('your-turn');
  }
}

// ── Status management ──
function setStatus(state) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  const thinking = document.getElementById('thinking');

  dot.className       = 'status-dot';
  thinking.className  = 'thinking-indicator';

  if (state === 'your-turn') {
    text.textContent = 'Your turn';
    dot.style.background = 'var(--green)';
  } else if (state === 'thinking') {
    text.textContent = 'Bot is thinking...';
    dot.classList.add('thinking');
    thinking.classList.add('active');
  } else if (state === 'gameover') {
    text.textContent = 'Game over';
    dot.classList.add('gameover');
  }
}

// ── Show result banner ──
function showResult(result, reason) {
  const banner  = document.getElementById('result-banner');
  const icon    = document.getElementById('result-icon');
  const text    = document.getElementById('result-text');
  const sub     = document.getElementById('result-sub');

  banner.style.display = 'flex';

  if (result === '1-0') {
    icon.textContent = '🎉';
    text.textContent = 'You won!';
  } else if (result === '0-1') {
    icon.textContent = '🤖';
    text.textContent = 'Bot wins!';
  } else {
    icon.textContent = '🤝';
    text.textContent = 'Draw!';
  }

  sub.textContent = reason || '';
}

// ── Move history ──
function updateMoveList() {
  const el = document.getElementById('move-list');

  if (moveHistory.length === 0) {
    el.innerHTML = '<span class="no-moves">No moves yet</span>';
    return;
  }

  // Group by move number
  const pairs = {};
  for (const m of moveHistory) {
    if (!pairs[m.moveNum]) pairs[m.moveNum] = {};
    if (m.isWhite) pairs[m.moveNum].white = m.uci;
    else           pairs[m.moveNum].black = m.uci;
  }

  el.innerHTML = Object.entries(pairs).map(([num, pair]) => `
    <span class="move-entry white">${num}. ${pair.white || ''}</span>
    ${pair.black ? `<span class="move-entry black">${pair.black}</span>` : ''}
  `).join('');

  el.scrollTop = el.scrollHeight;
}

// ── Reset game ──
async function resetGame() {
  const res  = await fetch('/reset', { method: 'POST' });
  const data = await res.json();

  fen            = data.fen;
  selectedSquare = null;
  legalMoves     = [];
  lastMove       = [];
  moveHistory    = [];
  moveCount      = 0;
  gameOver       = false;
  inCheckSquare  = null;

  document.getElementById('result-banner').style.display = 'none';
  document.getElementById('move-list').innerHTML =
    '<span class="no-moves">No moves yet</span>';

  setStatus('your-turn');
  renderBoard();
}

// ── Flip board ──
function flipBoard() {
  flipped = !flipped;
  renderBoard();
}

// ── Set personality ──
function setPersonality(p) {
  personality = p;

  // Update buttons
  document.querySelectorAll('.personality-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.personality === p);
  });

  // Update bot name and ELO in player info
  const info = {
    balanced:   { name: 'Balanced Bot',   elo: '~900 ELO' },
    greedy:     { name: 'Greedy Bot',     elo: '~800 ELO' },
    defensive:  { name: 'Defensive Bot',  elo: '~850 ELO' },
    aggressive: { name: 'Aggressive Bot', elo: '~950 ELO' },
  };

  document.getElementById('bot-name').textContent = info[p].name;
  document.getElementById('bot-elo').textContent  = info[p].elo;
}

// ── Init ──
renderBoard();
setStatus('your-turn');