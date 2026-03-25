import { useEffect, useRef } from "react";
import { Button, Box } from "@mui/material";

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  start(): void;
  stop(): void;
  abort(): void;
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
}

interface SpeechRecognitionEvent extends Event {
  readonly results: SpeechRecognitionResultList;
}

declare global {
  interface Window {
    webkitSpeechRecognition: {
      new (): SpeechRecognition;
    };
    SpeechRecognition: {
      new (): SpeechRecognition;
    };
  }
}


interface VoiceInputProps {
  onTranscriptChange: (text: string) => void;
  onListeningChange?: (listening: boolean) => void;
}

const VoiceInput: React.FC<VoiceInputProps> = ({
  onTranscriptChange,
  onListeningChange,
}) => {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const listeningRef = useRef(false);

  useEffect(() => {
    type SpeechRecognitionConstructor = new () => SpeechRecognition;

    const SpeechRecognitionConstructor =
      (
        window as {
          SpeechRecognition?: SpeechRecognitionConstructor;
          webkitSpeechRecognition?: SpeechRecognitionConstructor;
        }
      ).SpeechRecognition ||
      (
        window as {
          SpeechRecognition?: SpeechRecognitionConstructor;
          webkitSpeechRecognition?: SpeechRecognitionConstructor;
        }
      ).webkitSpeechRecognition;

    if (!SpeechRecognitionConstructor) {
      alert("이 브라우저는 SpeechRecognition을 지원하지 않습니다.");
      return;
    }

    const recognition = new SpeechRecognitionConstructor();
    recognition.lang = "ko-KR";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
      listeningRef.current = true;
      onListeningChange?.(true);
    };
    recognition.onend = () => {
      listeningRef.current = false;
      onListeningChange?.(false);
    };
    recognition.onerror = (event) => console.error("음성 인식 오류:", event);

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const result = Array.from(event.results)
        .map((res) => res[0].transcript)
        .join("");
      onTranscriptChange(result);
    };

    recognitionRef.current = recognition;
  }, [onTranscriptChange, onListeningChange]);

  const handleToggle = () => {
    if (!recognitionRef.current) return;

    if (listeningRef.current) {
      recognitionRef.current.stop();
    } else {
      onTranscriptChange("");
      recognitionRef.current.start();
    }
  };

  return (
    <Box textAlign="center" mt={4}>
      <Button variant="contained" onClick={handleToggle}>
        {listeningRef.current ? "중지" : "음성 입력 시작"}
      </Button>
    </Box>
  );
};

export default VoiceInput;
