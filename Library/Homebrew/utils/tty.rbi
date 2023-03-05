# typed: strict

module Tty
  sig { returns(String) }
  def self.blue; end

  sig { returns(String) }
  def self.bold; end

  sig { returns(String) }
  def self.cyan; end

  sig { returns(String) }
  def self.default; end

  sig { returns(String) }
  def self.down; end

  sig { returns(String) }
  def self.erase_char; end

  sig { returns(String) }
  def self.erase_line; end

  sig { returns(String) }
  def self.green; end

  sig { returns(String) }
  def self.italic; end

  sig { returns(String) }
  def self.left; end

  sig { returns(String) }
  def self.magenta; end

  sig { returns(String) }
  def self.no_underline; end

  sig { returns(String) }
  def self.red; end

  sig { returns(String) }
  def self.reset; end

  sig { returns(String) }
  def self.right; end

  sig { returns(String) }
  def self.strikethrough; end

  sig { returns(String) }
  def self.underline; end

  sig { returns(String) }
  def self.up; end

  sig { returns(String) }
  def self.yellow; end
end
