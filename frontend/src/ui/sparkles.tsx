import React, { useEffect, useRef } from 'react';

interface SparklesProps {
  id: string;
  background?: string;
  minSize?: number;
  maxSize?: number;
  particleDensity?: number;
  className?: string;
  particleColor?: string;
}

export const SparklesCore: React.FC<SparklesProps> = ({
  id,
  background = 'transparent',
  minSize = 0.6,
  maxSize = 1.4,
  particleDensity = 100,
  className = '',
  particleColor = '#FFFFFF',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    interface Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
      fadeSpeed: number;
    }

    const particles: Particle[] = [];

    // Create particles
    for (let i = 0; i < particleDensity; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 2,
        vy: (Math.random() - 0.5) * 2,
        size: Math.random() * (maxSize - minSize) + minSize,
        opacity: Math.random() * 0.5 + 0.5,
        fadeSpeed: Math.random() * 0.01 + 0.005,
      });
    }

    // Parse particle color
    const hexToRgb = (hex: string) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result
        ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16),
          }
        : { r: 255, g: 255, b: 255 };
    };

    const rgb = hexToRgb(particleColor);

    const animate = () => {
      // Clear canvas
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Update and draw particles
      particles.forEach((particle) => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.opacity -= particle.fadeSpeed;

        // Reset particle if it goes off screen or fades out
        if (particle.x < 0 || particle.x > canvas.width || particle.y < 0 || particle.y > canvas.height || particle.opacity <= 0) {
          particle.x = Math.random() * canvas.width;
          particle.y = Math.random() * canvas.height;
          particle.opacity = Math.random() * 0.5 + 0.5;
          particle.vx = (Math.random() - 0.5) * 2;
          particle.vy = (Math.random() - 0.5) * 2;
        }

        // Draw particle with glow effect
        ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${particle.opacity})`;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();

        // Add glow
        ctx.strokeStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${particle.opacity * 0.5})`;
        ctx.lineWidth = particle.size * 0.5;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size + 2, 0, Math.PI * 2);
        ctx.stroke();
      });

      requestAnimationFrame(animate);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [background, minSize, maxSize, particleDensity, particleColor]);

  return <canvas ref={canvasRef} id={id} className={className} />;
};
