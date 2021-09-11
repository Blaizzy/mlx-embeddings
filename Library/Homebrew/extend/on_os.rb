# typed: strict
# frozen_string_literal: true

module OnOS
  extend T::Sig

  # Block only executed on macOS. No-op on Linux.
  # <pre>on_macos do
  # # Do something Mac-specific
  # end</pre>
  sig { params(block: T.proc.void).void }
  def on_macos(&block)
    raise "No block content defined for 'on_macos' block" unless T.unsafe(block)
  end

  # Block only executed on Linux. No-op on macOS.
  # <pre>on_linux do
  # # Do something Linux-specific
  # end</pre>
  sig { params(block: T.proc.void).void }
  def on_linux(&block)
    raise "No block content defined for 'on_linux' block" unless T.unsafe(block)
  end
end

require "extend/os/on_os"
