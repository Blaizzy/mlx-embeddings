# typed: true
# frozen_string_literal: true

require_relative "directory"

module UnpackStrategy
  # Strategy for unpacking Git repositories.
  class Git < Directory
    using Magic

    def self.can_extract?(path)
      super && (path/".git").directory?
    end
  end
end
