# typed: true
# frozen_string_literal: true

require "utils/link"
require "settings"

# Helper functions for generating shell completions.
#
# @api private
module Completions
  extend T::Sig

  module_function

  sig { void }
  def link!
    Settings.write :linkcompletions, true
    Tap.each do |tap|
      Utils::Link.link_completions tap.path, "brew completions link"
    end
  end

  sig { void }
  def unlink!
    Settings.write :linkcompletions, false
    Tap.each do |tap|
      next if tap.official?

      Utils::Link.unlink_completions tap.path
    end
  end

  sig { returns(T::Boolean) }
  def link_completions?
    Settings.read(:linkcompletions) == "true"
  end

  sig { returns(T::Boolean) }
  def completions_to_link?
    shells = %w[bash fish zsh]
    Tap.each do |tap|
      next if tap.official?

      shells.each do |shell|
        return true if (tap.path/"completions/#{shell}").exist?
      end
    end

    false
  end

  sig { void }
  def show_completions_message_if_needed
    return if Settings.read(:completionsmessageshown) == "true"
    return unless completions_to_link?

    T.unsafe(self).ohai "Homebrew completions for external commands are unlinked by default!"
    T.unsafe(self).puts <<~EOS
      To opt-in to automatically linking Homebrew shell competion files, run:
        brew completions link
      Then, follow the directions at #{Formatter.url("https://docs.brew.sh/Shell-Completion")}
    EOS

    Settings.write :completionsmessageshown, true
  end
end
