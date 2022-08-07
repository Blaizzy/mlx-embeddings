# typed: true
# frozen_string_literal: true

module Homebrew
  # Helper module for simulating different system condfigurations.
  #
  # @api private
  class SimulateSystem
    class << self
      extend T::Sig

      attr_reader :arch

      sig { returns(T.nilable(Symbol)) }
      def os
        return :macos if @os.blank? && !OS.mac? && Homebrew::EnvConfig.simulate_macos_on_linux?

        @os
      end

      sig { params(new_os: Symbol).void }
      def os=(new_os)
        os_options = [:macos, :linux, *MacOSVersions::SYMBOLS.keys]
        raise "Unknown OS: #{new_os}" unless os_options.include?(new_os)

        @os = new_os
      end

      sig { params(new_arch: Symbol).void }
      def arch=(new_arch)
        raise "New arch must be :arm or :intel" unless [:arm, :intel].include?(new_arch)

        @arch = new_arch
      end

      sig { void }
      def clear
        @os = @arch = nil
      end

      sig { returns(T::Boolean) }
      def simulating_or_running_on_macos?
        return OS.mac? if os.blank?

        [:macos, *MacOSVersions::SYMBOLS.keys].include?(os)
      end

      sig { returns(T::Boolean) }
      def simulating_or_running_on_linux?
        return OS.linux? if os.blank?

        os == :linux
      end

      sig { returns(Symbol) }
      def current_arch
        @arch || Hardware::CPU.type
      end

      sig { returns(Symbol) }
      def current_os
        return T.must(os) if os.present?
        return :linux if OS.linux?

        MacOS.version.to_sym
      end
    end
  end
end
