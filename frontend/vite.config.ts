import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({plugins:[react(),tailwindcss()],build:{assetsDir:'web-assets'},server:{proxy:{'/api':'http://127.0.0.1:8000','/assets':'http://127.0.0.1:8000','/project-docs':'http://127.0.0.1:8000','/ready':'http://127.0.0.1:8000'}}});
