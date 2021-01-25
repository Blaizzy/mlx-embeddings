# typed: true
# frozen_string_literal: true

module OnOS
  # Block only executed on macOS. No-op on Linux.
  # <pre>on_macos do
  # # Do something Mac-specific
  # end</pre>
  def on_macos(&block)
    raise "No block content defined for 'on_macos' block" unless block
  end

  # Block only executed on Linux. No-op on macOS.
  # <pre>on_linux do
  # # Do something Linux-specific
  # end</pre>
  def on_linux(&block)
    raise "No block content defined for 'on_linux' block" unless block
  end
end

require "extend/os/on_os"
