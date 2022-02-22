# typed: true
# frozen_string_literal: true

class Formula
  extend OnOS

  def on_macos(&block)
    odeprecated "`on_macos do` inside `Formula` methods", "`if OS.mac?`"
    super
  end

  def on_linux(&block)
    odeprecated "`on_linux do` inside `Formula` methods", "`if OS.linux?`"
    super
  end

  extend Enumerable

  def self.each(&_block)
    # TODO: 3.4.0: odeprecated "`Enumerable` methods on `Formula`",
    #              "`Formula.all` (but avoid looping over all formulae, it's slow and insecure)"

    files.each do |file|
      yield Formulary.factory(file)
    rescue FormulaUnavailableError, FormulaUnreadableError => e
      # Don't let one broken formula break commands. But do complain.
      onoe "Failed to import: #{file}"
      $stderr.puts e
      next
    end
  end
end
