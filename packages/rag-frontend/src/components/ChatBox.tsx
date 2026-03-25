// ChatBox.tsx (수정본)
import { useState, useEffect, useRef } from "react";
import type { ChangeEvent } from "react";
import {
  Box,
  Button,
  Paper,
  TextField,
  Typography,
  CircularProgress,
  Collapse,
  Stack,
  Divider,
} from "@mui/material";
import { generate } from "../api";
import type { DocumentChunk, GenerateRequest } from "../types";
import VoiceInput from "./VoiceInput";

function formatElapsed(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m > 0 ? `${m}분 ` : ""}${s}초`;
}

export default function ChatBox() {
  const [question, setQuestion] = useState("");
  const [prompt, setPrompt] = useState("");
  const [docs, setDocs] = useState<DocumentChunk[] | null>([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);

  // 🔊 음성 읽기 ON/OFF (로컬 저장)
  const [speakEnabled, setSpeakEnabled] = useState<boolean>(() => {
    const saved = localStorage.getItem("speakEnabled");
    return saved ? saved === "1" : false; // 기본 OFF
  });

  const [elapsed, setElapsed] = useState(0);
  const [finalElapsed, setFinalElapsed] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const latestTranscriptRef = useRef<string>("");

  // 설정 변경 시 저장
  useEffect(() => {
    localStorage.setItem("speakEnabled", speakEnabled ? "1" : "0");
    if (!speakEnabled) {
      // OFF로 바꾸면 즉시 중단
      try {
        window.speechSynthesis.cancel();
      } catch { /* empty */ }
    }
  }, [speakEnabled]);

  // 🔊 음성 출력 함수 (ON일 때만 동작)
  const speak = (text: string) => {
    if (!text || !speakEnabled) return;
    const synth = window.speechSynthesis;
    try {
      synth.cancel(); // 이전 읽기 중단
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "ko-KR";
      utterance.rate = 1;
      utterance.pitch = 1;
      synth.speak(utterance);
    } catch {
      // 브라우저가 지원 안 하거나 일시 에러일 수 있음 — 조용히 무시
    }
  };

  // answer가 바뀌면 자동 읽어주기 (ON일 때만)
  useEffect(() => {
    if (answer) speak(answer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [answer, speakEnabled]);

  // 언마운트 시 안전 정리
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      try {
        window.speechSynthesis.cancel();
      } catch { /* empty */ }
    };
  }, []);

  const handleAsk = async (forcedQuestion?: string) => {
    const q = (forcedQuestion ?? question).trim();
    if (!q) return;

    setAnswer("");
    setPrompt("");
    setDocs([]);

    setLoading(true);
    setElapsed(0);
    setFinalElapsed(null);
    startTimeRef.current = Date.now();

    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    try {
      const req: GenerateRequest = { query: q }; // ← 여기만 꼭 q로!
      const res = await generate(req);
      setDocs(Array.isArray(res.reference_documents) ? res.reference_documents : []);
      setPrompt(res.prompt);
      setAnswer(res.response);
    } finally {
      if (intervalRef.current) clearInterval(intervalRef.current);
      setFinalElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        width: "100vw",
        bgcolor: "#f3f4f6",
        py: 6,
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
      }}
    >
      <Paper
        elevation={4}
        sx={{
          width: "100%",
          maxWidth: 800,
          p: 4,
          borderRadius: 3,
          bgcolor: "#ffffff",
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            justifyContent: "space-between",
          }}
        >
          <Typography variant="h5" gutterBottom sx={{ m: 0 }}>
            🧠 RAG 챗봇
          </Typography>

          {/* 🔊 음성 읽기 토글 버튼 */}
          <Button
            size="small"
            variant={speakEnabled ? "contained" : "outlined"}
            onClick={() => setSpeakEnabled((v) => !v)}
          >
            {speakEnabled ? "🔊 음성 출력 ON" : "🔇 음성 출력 OFF"}
          </Button>
        </Box>

        {/* 🔊 음성 입력 */}
        <VoiceInput
          onTranscriptChange={(text) => {
            setQuestion(text);
            latestTranscriptRef.current = text;
          }}
          onListeningChange={(isListening) => {
            setListening(isListening);
            if (!isListening) {
              const finalText = latestTranscriptRef.current.trim();
              if (finalText) handleAsk(finalText);
            }
          }}
        />

        {/* 키보드 입력 */}
        <TextField
          label="질문을 입력하세요"
          variant="outlined"
          fullWidth
          value={question}
          onChange={(e: ChangeEvent<HTMLInputElement>) => setQuestion(e.target.value)}
          sx={{ mt: 2, mb: 2 }}
          disabled={listening}
        />

        {/* 가운데 정렬된 질문 버튼 */}
        <Box textAlign="center" mt={4}>
          <Button variant="contained" onClick={() => handleAsk()} disabled={loading}>
            {loading ? `답변 생성 중... (${elapsed}초)` : "질문하기"}
          </Button>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* 로딩 중 */}
        {loading && (
          <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", py: 6 }}>
            <Stack spacing={2} alignItems="center">
              <CircularProgress />
              <Typography color="text.secondary">
                답변 생성 중... {elapsed > 0 ? `(${elapsed}초 경과)` : ""}
              </Typography>
            </Stack>
          </Box>
        )}

        {/* 결과 */}
        <Collapse in={!loading && !!answer} timeout={300} unmountOnExit>
          <Box
            sx={{
              mt: 2,
              p: 2,
              bgcolor: "#f9fafb",
              borderRadius: 2,
              border: "1px solid #e0e0e0",
            }}
          >
            <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
              답변:
            </Typography>
            <Typography sx={{ whiteSpace: "pre-wrap" }}>{answer}</Typography>

            {finalElapsed !== null && (
              <Typography sx={{ mt: 2, color: "gray" }}>
                ⏱️ 답변 시간: {formatElapsed(finalElapsed)}
              </Typography>
            )}
          </Box>

          {docs && docs.length > 0 && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: "#f9fafb",
                borderRadius: 2,
                border: "1px solid #e0e0e0",
                overflow: "hidden",
              }}
            >
              <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 2 }}>
                📚 참고자료
              </Typography>
              {docs.map((doc, index) => (
                <Box key={index} sx={{ mb: 2, overflowWrap: "break-word" }}>
                  <Typography variant="body1" component="div">
                    <a
                      href={doc.url || "#"}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{
                        fontWeight: "bold",
                        textDecoration: "underline",
                        color: "#1976d2",
                        wordBreak: "break-word",
                      }}
                      onClick={(e) => {
                        if (!doc.url) e.preventDefault();
                      }}
                    >
                      {doc.title || "제목 없음"}
                    </a>
                  </Typography>
                  {doc.chunk_text && (
                    <Typography
                      variant="body2"
                      sx={{
                        mt: 0.5,
                        color: "text.secondary",
                        whiteSpace: "pre-wrap",
                        wordBreak: "break-word",
                      }}
                    >
                      {doc.chunk_text}
                    </Typography>
                  )}
                </Box>
              ))}
            </Box>
          )}

          {prompt && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: "#f9fafb",
                borderRadius: 2,
                border: "1px solid #e0e0e0",
              }}
            >
              <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
                Prompt:
              </Typography>
              <Typography sx={{ whiteSpace: "pre-wrap" }}>{prompt}</Typography>
            </Box>
          )}
        </Collapse>
      </Paper>
    </Box>
  );
}
