# typed: true
# frozen_string_literal: true

require "utils/user"

module Cask
  # Helper functions for interacting with the `Caskroom` directory.
  #
  # @api private
  module Caskroom
    extend T::Sig

    sig { returns(Pathname) }
    def self.path
      @path ||= HOMEBREW_PREFIX.join("Caskroom")
    end

    sig { void }
    def self.ensure_caskroom_exists
      return if path.exist?

      sudo = !path.parent.writable?

      if sudo && !ENV.key?("SUDO_ASKPASS") && $stdout.tty?
        ohai "Creating Caskroom directory: #{path}",
             "We'll set permissions properly so we won't need sudo in the future."
      end

      SystemCommand.run("/bin/mkdir", args: ["-p", path], sudo: sudo)
      SystemCommand.run("/bin/chmod", args: ["g+rwx", path], sudo: sudo)
      SystemCommand.run("/usr/sbin/chown", args: [User.current, path], sudo: sudo)
      SystemCommand.run("/usr/bin/chgrp", args: ["admin", path], sudo: sudo)
    end

    sig { params(config: T.nilable(Config)).returns(T::Array[Cask]) }
    def self.casks(config: nil)
      return [] unless path.exist?

      Pathname.glob(path.join("*")).sort.select(&:directory?).map do |path|
        token = path.basename.to_s

        if tap_path = CaskLoader.tap_paths(token).first
          CaskLoader::FromTapPathLoader.new(tap_path).load(config: config)
        elsif caskroom_path = Pathname.glob(path.join(".metadata/*/*/*/*.rb")).first
          CaskLoader::FromPathLoader.new(caskroom_path).load(config: config)
        else
          CaskLoader.load(token, config: config)
        end
      end
    end
  end
end
