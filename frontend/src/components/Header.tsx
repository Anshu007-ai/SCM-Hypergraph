import React, { useState } from 'react';
import { Navigation, Info, Zap, Menu, X } from 'lucide-react';

type PageType = 'home' | 'analysis' | 'model' | 'about' | 'visualization';

interface HeaderProps {
  currentPage: PageType;
  onNavigate: (page: PageType) => void;
}

export const Header: React.FC<HeaderProps> = ({ currentPage, onNavigate }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems: Array<{ id: PageType; label: string; icon: React.ReactNode }> = [
    { id: 'home', label: 'Dashboard', icon: <Navigation className="w-4 h-4" /> },
    { id: 'analysis', label: 'Analysis', icon: <Navigation className="w-4 h-4" /> },
    { id: 'visualization', label: 'Visualization', icon: <Zap className="w-4 h-4" /> },
    { id: 'model', label: 'Model', icon: <Info className="w-4 h-4" /> },
    { id: 'about', label: 'About', icon: <Info className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-gray-700 bg-black/80 backdrop-blur supports-[backdrop-filter]:bg-black/60">
      <div className="max-w-7xl mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="px-3 py-1.5 rounded-lg bg-gradient-to-r from-blue-500 to-purple-500 text-white font-bold text-sm">
              HT-HGNN
            </div>
            <p className="text-sm text-gray-400 hidden sm:inline">Supply Chain Risk Analysis</p>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1 bg-gray-900 rounded-lg p-1">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`flex items-center px-3 py-2 rounded transition-all duration-200 text-sm font-medium whitespace-nowrap ${
                  currentPage === item.id
                    ? 'bg-gray-800 text-blue-400 shadow-sm'
                    : 'text-gray-400 hover:text-gray-300'
                }`}
              >
                {item.icon}
                <span className="ml-2">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 hover:bg-gray-800 rounded-lg transition-colors text-gray-400"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <nav className="md:hidden mt-3 space-y-1 pb-2 animate-in slide-in-from-top-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  onNavigate(item.id);
                  setMobileMenuOpen(false);
                }}
                className={`w-full text-left px-3 py-2 rounded transition-colors text-sm font-medium ${
                  currentPage === item.id
                    ? 'bg-gray-800 text-blue-400 font-semibold'
                    : 'text-gray-400 hover:bg-gray-800'
                }`}
              >
                <span className="flex items-center">
                  {item.icon}
                  <span className="ml-2">{item.label}</span>
                </span>
              </button>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
};
