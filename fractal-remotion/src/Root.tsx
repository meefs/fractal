import "./index.css";
import { Composition } from "remotion";
import { FractalRelease } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="FractalRelease"
        component={FractalRelease}
        durationInFrames={618}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
