# typed: strict

module Homebrew
  module CLI
    class Args < OpenStruct
      sig { returns(T.nilable(T::Boolean)) }
      def devel?; end

      sig { returns(T.nilable(T::Boolean)) }
      def HEAD?; end

      sig { returns(T.nilable(T::Boolean)) }
      def include_test?; end

      sig { returns(T.nilable(T::Boolean)) }
      def build_bottle?; end

      sig { returns(T.nilable(T::Boolean)) }
      def build_universal?; end

      sig { returns(T.nilable(T::Boolean)) }
      def build_from_source?; end

      sig { returns(T.nilable(T::Boolean)) }
      def force_bottle?; end

      sig { returns(T.nilable(T::Boolean)) }
      def debug?; end

      sig { returns(T.nilable(T::Boolean)) }
      def quiet?; end

      sig { returns(T.nilable(T::Boolean)) }
      def verbose?; end

      sig { returns(T.nilable(T::Boolean)) }
      def fetch_HEAD?; end

      sig { returns(T.nilable(T::Boolean)) }
      def cask?; end

      sig { returns(T.nilable(T::Boolean)) }
      def dry_run?; end

      sig { returns(T.nilable(T::Boolean)) }
      def skip_cask_deps?; end

      sig { returns(T.nilable(T::Boolean)) }
      def greedy?; end

      sig { returns(T.nilable(T::Boolean)) }
      def force?; end

      sig { returns(T.nilable(T::Boolean)) }
      def ignore_pinned?; end

      sig { returns(T.nilable(T::Boolean)) }
      def display_times?; end

      sig { returns(T.nilable(T::Boolean)) }
      def formula?; end
    end
  end
end
