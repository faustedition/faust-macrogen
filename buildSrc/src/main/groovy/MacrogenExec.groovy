import org.gradle.api.tasks.AbstractExecTask

class MacrogenExec extends AbstractExecTask<MacrogenExec> {

    def addInputArg(GString option, Object file) {
        this.args(option, file)
        this.inputs(file)
    }

    MacrogenExec() {
        super(MacrogenExec)

        group = 'macrogen'
        executable("$project.buildDir/envs/macrogen/bin/macrogen")

        if (project.rootProject != project) {
            addInputArg('--sigils', "$buildDir/sigils.json")
            addInputArg('--paralipomena', "$buildDir/www/data/paralipomena.js")
            args('--progressbar', 'false')
        }
    }

}
