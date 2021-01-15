# typed: strict
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def __caskroom_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--caskroom` [<cask>]

        Display Homebrew's Caskroom path.

        If <cask> is provided, display the location in the Caskroom where <cask>
        would be installed, without any sort of versioned directory as the last path.
      EOS

      named_args :cask
    end
  end

  sig { void }
  def __caskroom
    args = __caskroom_args.parse

    if args.named.to_casks.blank?
      puts Cask::Caskroom.path
    else
      args.named.to_casks.each do |cask|
        puts "#{Cask::Caskroom.path}/#{cask.token}"
      end
    end
  end
end
