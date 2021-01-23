# typed: strict

module Homebrew
  module CLI
    class Args < OpenStruct
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
      def newer_only?; end

      sig { returns(T.nilable(T::Boolean)) }
      def full_name?; end

      sig { returns(T.nilable(T::Boolean)) }
      def json?; end

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

      sig { returns(T.nilable(T::Boolean)) }
      def zap?; end

      sig { returns(T.nilable(T::Boolean)) }
      def ignore_dependencies?; end

      sig { returns(T.nilable(T::Boolean)) }
      def aliases?; end

      sig { returns(T.nilable(T::Boolean)) }
      def fix?; end

      sig { returns(T.nilable(T::Boolean)) }
      def keep_tmp?; end

      sig { returns(T.nilable(T::Boolean)) }
      def overwrite?; end

      sig { returns(T.nilable(T::Boolean)) }
      def silent?; end

      sig { returns(T.nilable(T::Boolean)) }
      def repair?; end

      sig { returns(T.nilable(T::Boolean)) }
      def prune_prefix?; end

      sig { returns(T.nilable(T::Boolean)) }
      def upload?; end

      sig { returns(T.nilable(T::Boolean)) }
      def total?; end

      sig { returns(T.nilable(T::Boolean)) }
      def dependents?; end

      sig { returns(T.nilable(T::Boolean)) }
      def installed?; end

      sig { returns(T.nilable(T::Boolean)) }
      def all?; end

      sig { returns(T.nilable(T::Boolean)) }
      def full?; end

      sig { returns(T.nilable(T::Boolean)) }
      def list_pinned?; end

      sig { returns(T.nilable(T::Boolean)) }
      def display_cop_names?; end

      sig { returns(T.nilable(T::Boolean)) }
      def syntax?; end

      sig { returns(T.nilable(T::Boolean)) }
      def ignore_non_pypi_packages?; end

      sig { returns(T.nilable(T::Boolean)) }
      def test?; end

      sig { returns(T.nilable(T::Boolean)) }
      def reverse?; end

      sig { returns(T.nilable(T::Boolean)) }
      def print_only?; end

      sig { returns(T.nilable(T::Boolean)) }
      def markdown?; end

      sig { returns(T.nilable(T::Boolean)) }
      def reset_cache?; end

      sig { returns(T.nilable(T::Boolean)) }
      def major?; end

      sig { returns(T.nilable(T::Boolean)) }
      def minor?; end

      sig { returns(T.nilable(String)) }
      def tag; end

      sig { returns(T.nilable(String)) }
      def tap; end

      sig { returns(T.nilable(String)) }
      def macos; end

      sig { returns(T.nilable(T::Array[String])) }
      def hide; end

      sig { returns(T.nilable(String)) }
      def version; end

      sig { returns(T.nilable(String)) }
      def name; end

      sig { returns(T.nilable(T::Boolean)) }
      def no_publish?; end

      sig { returns(T.nilable(T::Boolean)) }
      def shallow?; end

      sig { returns(T.nilable(T::Boolean)) }
      def fail_if_not_changed?; end

      sig { returns(T.nilable(String)) }
      def limit; end

      sig { returns(T.nilable(String)) }
      def message; end

      sig { returns(T.nilable(String)) }
      def issue; end

      sig { returns(T.nilable(String)) }
      def workflow; end

      sig { returns(T.nilable(String)) }
      def bintray_org; end

      sig { returns(T.nilable(String)) }
      def bintray_repo; end
      sig { returns(T.nilable(String)) }
      def package_name; end

      sig { returns(T.nilable(String)) }
      def prune; end

      sig { returns(T.nilable(T::Array[String])) }
      def only_cops; end

      sig { returns(T.nilable(T::Array[String])) }
      def except_cops; end

      sig { returns(T.nilable(T::Array[String])) }
      def only; end

      sig { returns(T.nilable(T::Array[String])) }
      def except; end

      sig { returns(T.nilable(T::Array[String])) }
      def mirror; end

      sig { returns(T.nilable(T::Array[String])) }
      def without_labels; end

      sig { returns(T.nilable(T::Array[String])) }
      def workflows; end

      sig { returns(T.nilable(T::Array[String])) }
      def ignore_missing_artifacts; end

      sig { returns(T.nilable(T::Array[String])) }
      def language; end

      sig { returns(T.nilable(T::Array[String])) }
      def extra_packages; end

      sig { returns(T.nilable(T::Array[String])) }
      def exclude_packages; end

      sig { returns(T.nilable(T::Array[String])) }
      def update; end

      sig { returns(T.nilable(T::Boolean)) }
      def s?; end

      sig { returns(T.nilable(String)) }
      def appdir; end

      sig { returns(T.nilable(String)) }
      def fontdir; end

      sig { returns(T.nilable(String)) }
      def colorpickerdir; end

      sig { returns(T.nilable(String)) }
      def prefpanedir; end

      sig { returns(T.nilable(String)) }
      def qlplugindir; end

      sig { returns(T.nilable(String)) }
      def dictionarydir; end

      sig { returns(T.nilable(String)) }
      def servicedir; end

      sig { returns(T.nilable(String)) }
      def input_methoddir; end

      sig { returns(T.nilable(String)) }
      def mdimporterdir; end

      sig { returns(T.nilable(String)) }
      def internet_plugindir; end

      sig { returns(T.nilable(String)) }
      def audio_unit_plugindir; end

      sig { returns(T.nilable(String)) }
      def vst_plugindir; end

      sig { returns(T.nilable(String)) }
      def vst3_plugindir; end

      sig { returns(T.nilable(String)) }
      def screen_saverdir; end
    end
  end
end
