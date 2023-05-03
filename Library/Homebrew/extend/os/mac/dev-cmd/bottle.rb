# typed: true
# frozen_string_literal: true

module Homebrew
  sig { returns(T::Array[String]) }
  def self.tar_args
    if MacOS.version >= :catalina
      ["--no-mac-metadata", "--no-acls", "--no-xattrs"].freeze
    else
      [].freeze
    end
  end
end
